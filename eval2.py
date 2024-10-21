
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

from mmcv import Config
from mmseg.utils import get_root_logger

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

class HelperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image, lidar2img, img_shape):
        
        img_metas = []
        img_metas.append(
            {"lidar2img": lidar2img, 
            "img_shape": img_shape}
        ) 
        out_ = self.model(img=image, img_metas=img_metas)
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
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "20506")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    print(f"tcp://{ip}:{port}")
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", 
        world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    world_size = dist.get_world_size()
    cfg.gpu_ids = range(world_size)
    torch.cuda.set_device(local_rank)

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

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
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model # .cuda()
    print('done ddp model')

    hp_model = HelperModel(my_model)

    image = torch.randn([1, 1, 3, 100, 100])# .cuda()
    lidar2img = np.random.randn(1, 1, 4, 4).tolist()
    print(lidar2img)
    img_shape =  [[100, 100]]
    all_input = (image, lidar2img, img_shape)

    # o_ = hp_model(image=image, lidar2img=lidar2img, img_shape=img_shape)
    # print(o_)

    torch.onnx.export(hp_model, all_input, "tpv.onnx", verbose=False, input_names=['input0', "input1", "input2"],
                      output_names=['output'], opset_version=11)
    
if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--ckpt-path', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
