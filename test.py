import argparse

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import *

from models.Mamba_Unet.vision_mamba import MambaUnet
from models.resm_mamba.resm_mamba import Resm_mamba
from models.resm_mamba2.resm_mamba2 import Resm_mamba2
from models.HMMamba.HMMamba import Resm_mamba3
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.dataset import RandomGenerator

from models.vmunet.vmunet import VMUNet

import os
import sys
import time
from tqdm import tqdm
# print("Available GPU devices:", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

from utils import *
from configs.config_setting_synapse import setting_config

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "0, 1, 2, 3"


def val_one_epoch(test_datasets,
                  test_loader,
                  model,
                  logger,
                  epoch,
                  config,
                  test_save_path,
                  val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes,
                                          patch_size=[config.input_size_h, config.input_size_w],
                                          test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing,
                                          val_or_test=val_or_test)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                                                                      np.mean(metric_i, axis=0)[0],
                                                                      np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime - stime:.2f}'
        print(log_info)
        logger.info(log_info)

    return performance, mean_hd95

def main(config):
    print('#----------Creating logger----------#')
    work_dir = 'results/MambaUnet_synapse_Saturday_16_November_2024_22h_26m_08s/'
    sys.path.append(work_dir + '/')
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
    outputs = os.path.join(work_dir, 'outputs')
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

    print('#----------Preparing dataset----------#')


    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
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









    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        epoch=0
        best_weight = torch.load(work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        mean_dice, mean_hd95 = val_one_epoch(
            val_dataset,
            val_loader,
            model,
            logger,
            epoch,
            config,
            test_save_path=outputs,
            val_or_test=True

        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir,
                         f'mean_dice{mean_dice:.4f}-mean_hd95{mean_hd95:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)