# 3DUNet patch training
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
import torch
import numpy as np
import SimpleITK as sitk
import math
from torch.utils.data import Dataset, DataLoader
crop_size = [160, 64, 64] # z, y, x
crop_stride = [80, 32, 32]
img_size = [441, 557, 557]

def slide_window(data, label, batchsize):
    a, b, c = crop_size[0], crop_size[1], crop_size[2]
    x, y, z = crop_stride[0], crop_stride[1], crop_stride[2] #步长
    l, w, h = data.shape[0], data.shape[1], data.shape[2]

    l_patchs, w_patchs, h_patchs = math.ceil((l - a) / x), math.ceil((w - b) / y), math.ceil((h - c) / z) # 567

    patchs = l_patchs * w_patchs * h_patchs

    if patchs % batchsize != 0:
        patchs += (batchsize - patchs % batchsize)

    dataset = np.zeros([patchs, 1, a, b, c])
    labelset = np.zeros([patchs, 1, a, b, c])
    sort = 0
    num_zero = 0
    for i in range(l_patchs):
        for j in range(w_patchs):
            for k in range(h_patchs):
                l_min = i * x
                l_max = i * x + a
                w_min = j * y
                w_max = j * y + b
                h_min = k * z
                h_max = k * z + c
                if i == l_patchs - 1:
                    l_min = l - a - 1
                    l_max = l - 1
                if j == w_patchs - 1:
                    w_min = w - b - 1
                    w_max = w - 1
                if k == h_patchs - 1:
                    h_min = h - c - 1
                    h_max = h - 1

                tmpdata = data[np.newaxis, l_min:l_max, w_min:w_max, h_min:h_max].astype(np.float32)
                tmplabel = label[np.newaxis, l_min:l_max, w_min:w_max, h_min:h_max].astype(np.float32)

                if np.sum(tmplabel) == 0:
                    num_zero += 1
                    continue
                dataset[sort] = tmpdata
                labelset[sort] = tmplabel

                sort += 1

    cal_patchs = patchs - num_zero
    if cal_patchs % batchsize != 0:
        cal_patchs += (batchsize - cal_patchs % batchsize)
    dataset = dataset[0:cal_patchs, :, :, :, :]
    labelset = labelset[0:cal_patchs, :, :, :, :]
    for i in range(sort, cal_patchs, 1):
        dataset[i] = dataset[sort - 1]
        labelset[i] = labelset[sort - 1]
    # print(dataset.shape)
    return dataset, labelset

class MyDataset(Dataset):
    def __init__(self, filepath, batchsize=4, stage='train'):
        self.filepath = filepath
        self.batchsize = batchsize
        self.stage = stage
        self.file_lists_ori = self.readpath(self.filepath)
        self.file_lists = []
        if stage == 'test':
            for file_name in self.file_lists_ori:
                file_num = int(file_name.split('.')[0])
                if file_num <= 87:
                    self.file_lists.append(file_name)
        else:
            self.file_lists = self.file_lists_ori

    # crop
    def __getitem__(self, index):
        data_path = '../../images/' + self.file_lists[index]
        lable_path = '../../segments/' + self.file_lists[index]

        raw_image = sitk.ReadImage(data_path)
        rescaleFilter = sitk.RescaleIntensityImageFilter()
        rescaleFilter.SetOutputMaximum(255)
        rescaleFilter.SetOutputMinimum(0)
        raw_image = rescaleFilter.Execute(raw_image)
        raw_image = sitk.Cast(raw_image, sitk.sitkFloat32)


        loaddata, loadlabel = sitk.GetArrayFromImage(raw_image) / 255, sitk.GetArrayFromImage(sitk.ReadImage(lable_path))

        if self.stage == 'test':
            data, label = loaddata[np.newaxis, :, :, :], loadlabel[np.newaxis, :, :, :]
            data, label = data.astype(np.float32), label.astype(np.float32) # data.shape: 1, a, b, c

        else:
            data, label = slide_window(loaddata, loadlabel, self.batchsize) # data.shape: patches, 1, a, b, c
        data, label = torch.from_numpy(data), torch.from_numpy(label)
        label = torch.clamp(label, min=0., max= 1.)
        return data, label, self.file_lists[index]

    def __len__(self):
        return len(self.file_lists)

    def readpath(self, path):
        file_lists = []
        with open(path, 'r') as file_load:
            while True:
                line = file_load.readline().strip()
                if not line: break
                file_lists.append(line)
        return file_lists