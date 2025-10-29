import argparse

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.Mamba_Unet.vision_mamba import MambaUnet
from models.resm_mamba.resm_mamba import Resm_mamba
from models.resm_mamba2.resm_mamba2 import Resm_mamba2
from models.HMMamba.HMMamba import Resm_mamba3
from models.resm_mamba4.resm_mamba4 import Resm_mamba4
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.dataset import RandomGenerator
from engine_ACDC import *
from config import *

from models.vmunet.vmunet import VMUNet

import os
import sys

# print("Available GPU devices:", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

from utils import *
from configs.config_setting_ACDC import setting_config

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # "0, 1, 2, 3"


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def main(config):
    print('#----------Creating logger----------#')
    work_dir = 'results/MambaUnet_synapse_Saturday_16_November_2024_22h_26m_08s/'
    sys.path.append(work_dir + '/')
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')

    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()
    gpus_type, gpus_num = torch.cuda.get_device_name(), torch.cuda.device_count()
    if config.distributed:
        print('#----------Start DDP----------#')
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(config.seed)
        config.local_rank = torch.distributed.get_rank()

    labeled_slice = patients_to_slices(config.data_path, 140)

    print('#----------Preparing dataset----------#')
    train_dataset = config.datasets(base_dir=config.data_path, num=labeled_slice, split="train",
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]))
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size // gpus_num if config.distributed else config.batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              num_workers=config.num_workers,
                              sampler=train_sampler)

    val_dataset = config.datasets(base_dir=config.data_path, split="val")
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                            batch_size=1,  # if config.distributed else config.batch_size,
                            shuffle=False,
                            pin_memory=False,
                            num_workers=0,
                            sampler=val_sampler,
                            drop_last=True)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
    elif config.network == 'resm_mamba3_conmcsm_HM2D':
        model = Resm_mamba3(num_classes=model_cfg['num_classes'],
                            input_channel=model_cfg['input_channels'],
                            drop_path_rate=0.3, )
    elif config.network=='MambaUnet':
        parser = argparse.ArgumentParser()
        parser.add_argument('--root_path', type=str,
                            default='../data/ACDC', help='Name of Experiment')
        parser.add_argument('--exp', type=str,
                            default='ACDC/Fully_Supervised', help='experiment_name')
        parser.add_argument('--model', type=str,
                            default='mambaunet', help='model_name')
        parser.add_argument('--num_classes', type=int, default=9,
                            help='output channel of network')

        parser.add_argument(
            '--cfg', type=str, default="./models/Mamba_Unet/vmamba_tiny.yaml", help='path to config file', )
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
        parser.add_argument('--zip', action='store_true',
                            help='use zipped dataset instead of folder dataset')
        parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                            help='no: no cache, '
                                 'full: cache all data, '
                                 'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
        parser.add_argument('--resume', help='resume from checkpoint')
        parser.add_argument('--accumulation-steps', type=int,
                            help="gradient accumulation steps")
        parser.add_argument('--use-checkpoint', action='store_true',
                            help="whether to use gradient checkpointing to save memory")
        parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true',
                            help='Perform evaluation only')
        parser.add_argument('--throughput', action='store_true',
                            help='Test throughput only')

        parser.add_argument('--max_iterations', type=int,
                            default=10000, help='maximum epoch number to train')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batch_size per gpu')
        parser.add_argument('--deterministic', type=int, default=1,
                            help='whether use deterministic training')
        parser.add_argument('--base_lr', type=float, default=0.01,
                            help='segmentation network learning rate')
        parser.add_argument('--patch_size', type=list, default=[224, 224],
                            help='patch size of network input')
        parser.add_argument('--seed', type=int, default=1337, help='random seed')
        parser.add_argument('--labeled_num', type=int, default=140,
                            help='labeled data')
        args = parser.parse_args()
        con = get_config(args)
        model= MambaUnet(con,num_classes=model_cfg['num_classes'],
                        )
    else:
        raise ('Please prepare a right net!')

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        # model = model.cuda()
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    # cal_params_flops(model, 224, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    # print('#----------Training----------#')
    # for epoch in range(start_epoch, config.epochs + 1):
    #
    #     torch.cuda.empty_cache()
    #     train_sampler.set_epoch(epoch) if config.distributed else None
    #
    #     loss = train_one_epoch(
    #         train_loader,
    #         model,
    #         criterion,
    #         optimizer,
    #         scheduler,
    #         epoch,
    #         logger,
    #         config,
    #         scaler=scaler
    #     )
    #
    #     if loss < min_loss:
    #         torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
    #         min_loss = loss
    #         min_epoch = epoch
    #     if 49 < epoch < 300 and epoch % 50 == 0:
    #         torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_' + str(epoch) + '.pth'))
    #
    #     if epoch % config.val_interval == 0:
    #         mean_dice, mean_hd95 = val_one_epoch(
    #             val_dataset,
    #             val_loader,
    #             model,
    #             epoch,
    #             logger,
    #             config,
    #             test_save_path=outputs,
    #             val_or_test=False
    #         )
    #
    #     torch.save(
    #         {
    #             'epoch': epoch,
    #             'min_loss': min_loss,
    #             'min_epoch': min_epoch,
    #             'loss': loss,
    #             'model_state_dict': model.module.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #         }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        epoch=0
        best_weight = torch.load(work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        mean_dice, mean_hd95 = val_one_epoch(
            val_dataset,
            val_loader,
            model,
            epoch,
            logger,
            config,
            test_save_path=outputs,
            val_or_test=True
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir,
                         f'best-epoch{min_epoch}-mean_dice{mean_dice:.4f}-mean_hd95{mean_hd95:.4f}.pth')
        )
    # model_list = [50,100, 150, 200, 250]
    # for i in model_list:
    #     model_name = 'best_' + str(i) + '.pth'
    #     if os.path.exists(os.path.join(checkpoint_dir, model_name)):
    #         print('#----------Testing----------#')
    #         best_weight = torch.load(config.work_dir + 'checkpoints/' + model_name, map_location=torch.device('cpu'))
    #         model.module.load_state_dict(best_weight)
    #         # model.load_state_dict(best_weight)
    #         mean_dice, mean_hd95 = val_one_epoch(
    #             val_dataset,
    #             val_loader,
    #             model,
    #             epoch,
    #             logger,
    #             config,
    #             test_save_path=outputs,
    #             val_or_test=False
    #         )
    #         os.rename(
    #             os.path.join(checkpoint_dir, model_name),
    #             os.path.join(checkpoint_dir,
    #                          'best_' + str(i) + f'-mean_dice{mean_dice:.4f}-mean_hd95{mean_hd95:.4f}.pth')
    #         )


if __name__ == '__main__':
    config = setting_config
    main(config)