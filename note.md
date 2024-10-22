pytorch-1.10.0 不支持算子 maximum, nan_to_num，grid_sampler

这些算子均在1.12.0被支持，最好能升级一下pytorch版本

maximum 替换成了 torch.clamp

nan_to_num 直接删除

grid_sampler 替换成  mmcv.ops.point_sample.bilinear_grid_sample (该算子在mmcv种的ops/multi_scale_deform_attn.py 也有也需要修改)


--------
另外 删除了所有的checkpoint函数
将所有register_buffer修改

-------

解决MMCVModulatedDeformConv2d算子的问题，按照如下文档操作即可（把之前安装的mmcv-full先卸载）

```
https://github.com/yangrisheng/mmcv/blob/master/docs/en/deployment/onnxruntime_op.md
```