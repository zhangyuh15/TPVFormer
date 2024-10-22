
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

from mmcv import Config
from mmseg.utils import get_root_logger
import onnxruntime as ort

import warnings
warnings.filterwarnings("ignore")

"""
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
"""

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

    # logger = get_root_logger(log_file=None, log_level='INFO')
    # # logger.info(f'Config:\n{cfg.pretty_text}')

    # # build model
    # if cfg.get('occupancy', False):
    #     from builder import tpv_occupancy_builder as model_builder
    # else:
    #     from builder import tpv_lidarseg_builder as model_builder
    
    # my_model = model_builder.build(cfg.model)
    # n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    # logger.info(f'Number of params: {n_parameters}')

    # my_model = my_model # .cuda()
    # print('done ddp model')



    # TODO: Are these input shape right, need to check 

    data_from_nuscene = np.load("samples/sample_0.npy",allow_pickle=True).item()
    imgs = data_from_nuscene["imgs"]
    img_metas = data_from_nuscene["img_metas"]
    val_vox_label = data_from_nuscene["val_vox_label"]
    val_grid = data_from_nuscene["val_grid"]
    val_pt_labs = data_from_nuscene["val_pt_labs"]

    imgs = imgs.cpu()# .cuda()
    B, N, C, H, W = imgs.size()
    imgs = imgs.reshape(B * N, C, H, W)
    val_grid_float = val_grid.to(torch.float32) # .cuda()
    val_pt_labs = val_pt_labs# .cuda()

    lidar2img = [xx.tolist() for xx in img_metas[0]["lidar2img"]]
    img_shape = img_metas[0]["img_shape"]

    import_options = True
    if import_options:
        from mmcv.ops import get_onnxruntime_op_path

    ort_custom_op_path = get_onnxruntime_op_path()
    print("ort_custom_op_path:", ort_custom_op_path)
    assert os.path.exists(ort_custom_op_path)

    ##################################
    # print("export: backbone")  # THis part is OK
    # all_input = (imgs, lidar2img, img_shape, val_grid_float)

    # session_options = ort.SessionOptions()
    # session_options.register_custom_ops_library(ort_custom_op_path)
    # if import_options:
    #     ort_session = ort.InferenceSession("debug_submodel/tpv_img_backbone.onnx", session_options)
    # else:
    #      ort_session = ort.InferenceSession("tpv_img_backbone.onnx")

    # inputs = {
    #     ort_session.get_inputs()[0].name: imgs.numpy().astype(np.float32), 
    #     }
    # start_time = time.time()
    # onnx_res = ort_session.run(None, inputs)  # This taks 38min
    # print("all time:", time.time() - start_time)

    # # print(torch_res1.shape, onnx_res.shape)
    
    #################################################
    # print("export: neck")
    # nect_ipt = np.load("debug_data/neck_ipt.npy", allow_pickle=True).item()["neck_ipt"]

    # print("-->", len(nect_ipt))
    # all_input = nect_ipt
    # session_options = ort.SessionOptions()
    # session_options.register_custom_ops_library(ort_custom_op_path)
    # ort_session = ort.InferenceSession("debug_submodel/tpv_img_neck.onnx", session_options)

    # inputs = {
    #     ort_session.get_inputs()[0].name: all_input[0].detach().numpy().astype(np.float32), 
    #     ort_session.get_inputs()[1].name: all_input[1].detach().numpy().astype(np.float32), 
    #     ort_session.get_inputs()[2].name: all_input[2].detach().numpy().astype(np.float32), 
    #     }
    
    # t1 = time.time()
    # onnx_res = ort_session.run(None, inputs)  # This taks 0.55s
    # t2 = time.time()

    # print(f"Convert time: {t2- t1} [s]")
    
    #########################################
    print("export: head")
    img_feats = np.load("debug_data/img_feats.npy", allow_pickle=True).item()["img_feats"]

    all_input = (img_feats, lidar2img, img_shape)
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(ort_custom_op_path)
    ort_session = ort.InferenceSession("debug_submodel/tpv_head.onnx", session_options)

    inputs = {
        ort_session.get_inputs()[0].name: all_input[0].detach().numpy().astype(np.float32), 
        ort_session.get_inputs()[1].name: np.array(all_input[1]).astype(np.float32), 
        ort_session.get_inputs()[2].name: np.array(all_input[2]).astype(np.int64), 
        }

    t1 = time.time()
    onnx_res = ort_session.run(None, inputs)
    t2 = time.time()
    ort_session.close()
    print(f"Convert time: {t2- t1} [s]")

    # #########################################
    # print("export: agg")
    # agg_ipt = np.load("debug_data/agg_ipt.npy", allow_pickle=True).item()["agg_ipt"]

    # all_input = (agg_ipt[0], agg_ipt[1], agg_ipt[2], val_grid_float)
    # session_options = ort.SessionOptions()
    # session_options.register_custom_ops_library(ort_custom_op_path)
    # ort_session = ort.InferenceSession("debug_submodel/tpv_agg.onnx", session_options)
    # inputs = {
    #     ort_session.get_inputs()[0].name: all_input[0].detach().numpy().astype(np.float32), 
    #     ort_session.get_inputs()[1].name: all_input[1].detach().numpy().astype(np.float32), 
    #     ort_session.get_inputs()[2].name: all_input[2].detach().numpy().astype(np.float32), 
    #     ort_session.get_inputs()[3].name: all_input[3].numpy().astype(np.float32), 

    #     }
    # t1 = time.time()
    # onnx_res = ort_session.run(None, inputs) # This takes 0.7535841464996338 s
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
