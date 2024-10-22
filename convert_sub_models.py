
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder
import time

from mmcv import Config
from mmseg.utils import get_root_logger

import warnings
warnings.filterwarnings("ignore")
import mmcv
from mmcv.onnx import register_extra_symbolics

opset_version = 13
register_extra_symbolics(opset_version)

def pass_print(*args, **kwargs):
    pass

class HelperModelBackbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image):
        out_ = self.model(image)
        return out_
    
class HelperModelNeck(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, t1, t2, t3):
        out_ = self.model((t1, t2, t3))
        return out_

class HelperModelHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image, lidar2img, img_shape):
        img_metas = []
        img_metas.append(
            {"lidar2img": lidar2img, 
            "img_shape": img_shape}
        ) 
        out_ = self.model(image, img_metas)
        return out_
    
class HelperModelAgg(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, ipt1,ipt2, ipt3, point):
        out_ = self.model([ipt1,ipt2, ipt3], point)
        return out_
    

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

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
    distributed = False

    logger = get_root_logger(log_file=None, log_level='INFO')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    
    my_model = model_builder.build(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    my_model = my_model # .cuda()
    print('done ddp model')



    # TODO: Are these input shape right, need to check 

    ## TODO Please provide some input data of this model with right shape and dtype

    data_from_nuscene = np.load("samples/sample_0.npy",allow_pickle=True).item()
    imgs = data_from_nuscene["imgs"]
    img_metas = data_from_nuscene["img_metas"]
    val_vox_label = data_from_nuscene["val_vox_label"]
    val_grid = data_from_nuscene["val_grid"]
    val_pt_labs = data_from_nuscene["val_pt_labs"]

    imgs = imgs.cpu()# .cuda()
    val_grid_float = val_grid.to(torch.float32) # .cuda()
    val_pt_labs = val_pt_labs# .cuda()

    # print(type(), type(img_metas[0]["lidar2img"][0]))
    # exit()
    lidar2img = [xx.tolist() for xx in img_metas[0]["lidar2img"]]
    img_shape = img_metas[0]["img_shape"]

    imgs = imgs.cpu()# .cuda()
    B, N, C, H, W = imgs.size()
    imgs = imgs.reshape(B * N, C, H, W)
    all_input = (imgs)
    
    #########################################
    # torch.save(hp_model.state_dict(), "tpv_cpu.pth")
    # print("export: backbone")
    # hp_model = HelperModelBackbone(my_model.img_backbone)
    # t1 = time.time()
    # torch.onnx.export(hp_model, all_input, "./tpv_img_backbone.onnx", verbose=False, input_names=['input0'],
    #                   output_names=['output0'], opset_version=13)
    # t2 = time.time()
    # print(f"Convert time: {t2- t1} [s]")

    # #########################################
    # print("export: neck")
    # nect_ipt = np.load("neck_ipt.npy", allow_pickle=True).item()["neck_ipt"]

    # print("-->", len(nect_ipt))
    # hp_model = HelperModelNeck(my_model.img_neck)
    # all_input = nect_ipt
    # t1 = time.time()
    # torch.onnx.export(hp_model, all_input, "./tpv_img_neck.onnx", verbose=False, input_names=['input0', 'input1', 'input2'],
    #                   output_names=['output0'], opset_version=13)
    # t2 = time.time()
    # print(f"Convert time: {t2- t1} [s]")

    # #########################################
    print("export: head")
    img_feats = np.load("debug_data/img_feats.npy", allow_pickle=True).item()["img_feats"]

    hp_model = HelperModelHead(my_model.tpv_head)
    all_input = (img_feats, lidar2img, img_shape)
    t1 = time.time()
    torch.onnx.export(hp_model, all_input, "debug_submodel/tpv_head.onnx", verbose=False, input_names=['input0', 'input1', 'input2'],
                      output_names=['output0'], opset_version=13)
    t2 = time.time()
    print(f"Convert time: {t2- t1} [s]")

    #########################################
    # print("export: agg")
    # agg_ipt = np.load("debug_data/agg_ipt.npy", allow_pickle=True).item()["agg_ipt"]

    # hp_model = HelperModelAgg(my_model.tpv_aggregator)
    # print(len(agg_ipt), type(agg_ipt))
    # all_input = (agg_ipt[0], agg_ipt[1], agg_ipt[2], val_grid_float)
    # t1 = time.time()
    # torch.onnx.export(hp_model, all_input, "debug_submodel/tpv_agg.onnx", verbose=False, input_names=['input0', 'input1', 'input2', 'input3'],
    #                   output_names=['output0', 'output1'], opset_version=13)
    # t2 = time.time()
    # print(f"Convert time: {t2- t1} [s]")

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--ckpt-path', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    main(0, args)
