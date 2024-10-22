
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist


from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

from config.tpv04_occupancy import dataset_params as dataset_config
from config._base_.dataset import train_data_loader as train_dataloader_config
from config._base_.dataset import val_data_loader as val_dataloader_config
from config.tpv04_occupancy import grid_size
import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main():

    # load config

    # check label_mapping, fill_label, ignore_label, pc_dataset_type
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']

    # check num_workers, imageset
    # train_dataloader_config = cfg.train_data_loader
    # val_dataloader_config = cfg.val_data_loader

    # init DDP


    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=False,
            scale_rate=1
        )

    os.makedirs("samples", exist_ok=True)
    with torch.no_grad():
        for i_iter_val, (imgs, img_metas, val_vox_label, val_grid, val_pt_labs) in enumerate(val_dataset_loader):
            
            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            vox_label = val_vox_label.type(torch.LongTensor).cuda()
            val_pt_labs = val_pt_labs.cuda()
            
            to_save = dict(
                imgs=imgs,
                img_metas=img_metas,
                val_vox_label=val_vox_label,
                val_grid=val_grid,
                val_pt_labs=val_pt_labs,
            )
            np.save("samples/sample_{}".format(i_iter_val), to_save)

            if i_iter_val > 10:
                break

            # predict_labels_vox, predict_labels_pts = my_model(img=imgs, img_metas=img_metas, points=val_grid_float)
            
        

if __name__ == '__main__':
    # Eval settings
    main()
