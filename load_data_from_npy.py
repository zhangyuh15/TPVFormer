import numpy as np
import torch

for i in range(10):
    data_from_nuscene = np.load(f"samples/sample_{i}.npy",allow_pickle=True).item()
    imgs = data_from_nuscene["imgs"]
    img_metas = data_from_nuscene["img_metas"]
    val_vox_label = data_from_nuscene["val_vox_label"]
    val_grid = data_from_nuscene["val_grid"]
    val_pt_labs = data_from_nuscene["val_pt_labs"]

    imgs = imgs # .cuda()
    val_grid_float = val_grid.to(torch.float32) # .cuda()
    val_grid_int = val_grid.to(torch.long) # .cuda()
    vox_label = val_vox_label.type(torch.LongTensor)# .cuda()
    val_pt_labs = val_pt_labs# .cuda()


    for tt in img_metas:
        print(i, " ", tt.keys())

    print("image_metas:", len(img_metas), img_metas[0].keys())
