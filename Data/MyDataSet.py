# whole image training
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
import torch
import random
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
crop_size = [64, 64, 128]
crop_stride = [32, 32, 64]
img_size = [557, 557, 441]

def crop_by_list(data, label):
    random_x_max = data.shape[0] - crop_size[0]
    random_y_max = data.shape[1] - crop_size[1]
    random_z_max = data.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)
    data = data[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                 z_random:z_random + crop_size[2]]

    return data, label

class MyDataset(Dataset):
    def __init__(self, split_file_path, stage):
        self.split_file_path = split_file_path
        self.stage = stage
        self.file_lists_ori = self.readpath(split_file_path)
        self.file_lists = []

        if stage == 'test':
            for file_name in self.file_lists_ori:
                file_num = int(file_name.split('.')[0])
                if file_num <= 87:
                    self.file_lists.append(file_name)
        else:
            self.file_lists = self.file_lists_ori
    def __getitem__(self, index):
        data_path = '../../images/' + self.file_lists[index]
        lable_path = '../../segments/' + self.file_lists[index]

        raw_image = sitk.ReadImage(data_path)
        rescaleFilter = sitk.RescaleIntensityImageFilter()
        rescaleFilter.SetOutputMaximum(255)
        rescaleFilter.SetOutputMinimum(0)
        raw_image = rescaleFilter.Execute(raw_image)

        data, label = sitk.GetArrayFromImage(raw_image) / 255, sitk.GetArrayFromImage(sitk.ReadImage(lable_path))

        data, label = data[np.newaxis, :, :, :], np.clip(label[np.newaxis, :, :, :].astype(np.uint8), 0, 1)
        data, label = torch.from_numpy(data).to(torch.float16), torch.from_numpy(label).to(torch.float16)
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