# dataset 4 resnet
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

pos_data_list = [
    '01.nii.gz', '02.nii.gz', '03.nii.gz', '04.nii.gz', '05.nii.gz', '06.nii.gz', '07.nii.gz', '08.nii.gz',
    '09.nii.gz', '10.nii.gz', '11.nii.gz', '12.nii.gz', '13.nii.gz', '14.nii.gz', '15.nii.gz',
    '16.nii.gz', '17.nii.gz', '18.nii.gz', '19.nii.gz', '20.nii.gz', '21.nii.gz', '22.nii.gz', '23.nii.gz',
    '24.nii.gz', '25.nii.gz', '26.nii.gz', '27.nii.gz', '28.nii.gz', '29.nii.gz', '30.nii.gz',
    '31.nii.gz', '32.nii.gz', '33.nii.gz', '34.nii.gz', '35.nii.gz', '36.nii.gz', '37.nii.gz', '38.nii.gz',
    '39.nii.gz', '40.nii.gz', '41.nii.gz', '42.nii.gz', '43.nii.gz', '44.nii.gz', '45.nii.gz',
    '46.nii.gz', '47.nii.gz', '48.nii.gz', '49.nii.gz', '50.nii.gz', '51.nii.gz', '52.nii.gz'
]
neg_data_list = [
    '53.nii.gz', '54.nii.gz', '55.nii.gz', '56.nii.gz', '57.nii.gz', '58.nii.gz', '59.nii.gz', '60.nii.gz',
    '61.nii.gz', '62.nii.gz', '63.nii.gz', '64.nii.gz', '65.nii.gz', '66.nii.gz', '67.nii.gz', '68.nii.gz', '69.nii.gz',
    '70.nii.gz', '71.nii.gz', '72.nii.gz', '73.nii.gz', '74.nii.gz', '75.nii.gz',
    '76.nii.gz', '77.nii.gz', '78.nii.gz', '79.nii.gz', '80.nii.gz', '81.nii.gz', '82.nii.gz', '83.nii.gz', '84.nii.gz',
    '85.nii.gz', '86.nii.gz', '87.nii.gz'
]


def resize_img(itk_img, new_spacing, origin_spacing, resampled_method=sitk.sitkNearestNeighbor):
    newSpacing = np.array(new_spacing, float)
    resampler = sitk.ResampleImageFilter()
    originSize = itk_img.GetSize()
    factor = newSpacing / origin_spacing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itk_img)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resampled_method)
    itkimgResampled = resampler.Execute(itk_img)
    return itkimgResampled


class MyDataset(Dataset):
    def __init__(self, rootpath, split_file_name, stage='train', resize=None, direct_get=False):
        self.rootpath = rootpath
        self.direct_get = direct_get
        self.stage = stage
        self.split_file_name = split_file_name
        self.resize = resize
        if direct_get:
            relative_path = '../../output/'
            direct_list = os.listdir(relative_path)
            for idx, (d) in enumerate(direct_list):
                direct_list[idx] = relative_path + d
            self.file_lists = direct_list
        else:
            split_file_path = os.path.join(self.rootpath, self.split_file_name)
            raw_file_lists = self.readpath(split_file_path)
            self.file_lists = raw_file_lists[:]

    def __getitem__(self, index):
        ori_file_len = 87
        data_path = self.file_lists[index]
        data_name = data_path.split(os.path.sep)[-1]
        data_num = int(data_name.split('.')[0])
        scale_data_num = data_num % ori_file_len
        if scale_data_num == 0:
            scale_data_num = ori_file_len
        label = None
        if self.direct_get:
            for pos in pos_data_list:
                pos_num = int(pos.split('.')[0])
                if scale_data_num == pos_num:
                    label = 0
                    break
            if label == None:
                label = 1
        else:
            label = int(data_path.split(os.sep)[-2])
        img = sitk.ReadImage(data_path)
        resized_img = sitk.GetArrayFromImage(img)
        data = resized_img[np.newaxis, :, :, :].astype(np.float32)  # (1, z, y, x)
        data = torch.from_numpy(data).to(torch.float32)
        label = torch.tensor(label).to(torch.long)
        return data, label, data_path.split(os.sep)[-1]

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
