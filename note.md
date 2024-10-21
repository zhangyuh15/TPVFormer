pytorch-1.10.0 不支持算法 maximum, nan_to_num，grid_sampler

这些算子均在1.12.0被支持，最好能升级一下pytorch版本

maximum 替换成了 torch.clamp

nan_to_num 直接删除

grid_sampler 替换成  mmcv.ops.point_sample.bilinear_grid_sample (该算子在mmcv种的ops/multi_scale_deform_attn.py 也有也需要修改)