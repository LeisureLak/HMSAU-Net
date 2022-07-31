
# diy dataloader for resnet training, padding side length to the multiple of n-th power of 2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(MyDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, data):
        img_list, label_list, name_list = zip(*data)
        assert len(img_list) == len(label_list)
        batch_size = len(img_list)
        pad_img_list = []

        l_list = [int(s.shape[1]) for s in img_list]
        w_list = [int(s.shape[2]) for s in img_list]
        h_list = [int(s.shape[3]) for s in img_list]
        max_l, max_w, max_h = np.array(l_list).max(), np.array(w_list).max(), np.array(h_list).max()
        for i in range(batch_size):
            img = img_list[i][0]
            pad_img = F.pad(img, (0, int(max_h - img.shape[2]), 0, int(max_w - img.shape[1]), 0, int(max_l - img.shape[0])), value=0.)
            pad_img_list.append(torch.unsqueeze(pad_img, dim=0))
        batch_img = torch.stack(pad_img_list)
        batch_label = torch.stack(label_list)
        return batch_img, batch_label, name_list