
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

from mmcv import Config

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):

    # load config
    cfg = Config.fromfile(args.py_config)

    # check label_mapping, fill_label, ignore_label, pc_dataset_type
    dataset_config = cfg.dataset_params
    
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    # check num_workers, imageset
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    grid_size = cfg.grid_size

    # init DDP


    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1)
        )

    with torch.no_grad():
        for i_iter_val, (imgs, img_metas, val_vox_label, val_grid, val_pt_labs) in enumerate(val_dataset_loader):
            
            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            vox_label = val_vox_label.type(torch.LongTensor).cuda()
            val_pt_labs = val_pt_labs.cuda()

            # predict_labels_vox, predict_labels_pts = my_model(img=imgs, img_metas=img_metas, points=val_grid_float)
            
        

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--ckpt-path', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
