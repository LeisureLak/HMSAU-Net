# -*- codeing = utf-8 -*-
# @Time : 2021/6/5 3:52 下午
# @Author : 安康
# @File : utils.py
# @Software : PyCharm
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
import Config
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from scipy import ndimage
import random
from Net import UNet3D
from Net import SAUNet
from Net import Resnet
import SimpleITK as itk

device = torch.device('cuda:0')

args = Config.args

# 调整学习率
def adjust_learning_rate(optimizer, epoch, lr=None):
    if lr == None:
        lr = args.lr_crop * (0.1 ** (epoch // 50))
    else:
        lr = lr * (0.5 ** (epoch // 100))
    # lr = 0.00005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 立刻调整学习率
def immediately_adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.5


# 获取last resnet分类模型
def load_model_resnet(device, GPU, layers, model_url):
    print('loading old model...')

    # 获取训练好的模型
    model = resnet.generate_model(layers, n_classes=1, n_input_channels=1).to(device)
    model_config = torch.load(model_url, map_location='cuda:' + GPU)
    model.load_state_dict(model_config['net'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(model_config['optimizer'])
    last_epoch = model_config['epoch']
    best_val_acc = model_config['best_val_acc']
    best_val_epoch = model_config['best_val_epoch']

    return model, optimizer, last_epoch, best_val_acc, best_val_epoch


# 获取last unet模型
def load_model_unet(device, GPU, model_path):
    print('loading old model...')

    # 获取训练好的模型
    model = UNet3D.UNet3D()
    model_config = torch.load(model_path, map_location='cuda:' + GPU)
    model.load_state_dict(model_config['net'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(model_config['optimizer'])
    last_epoch = model_config['epoch']
    best_val_loss = model_config['best_val_loss']
    if best_val_loss > 1:
        best_val_loss = 0
    best_val_epoch = model_config['best_val_epoch']

    return model, optimizer, last_epoch, best_val_loss, best_val_epoch


# 获取last saunet模型
def load_model_saunet(device, GPU, model_path):
    print('loading old model...')

    # 获取训练好的模型
    model = saunet.SAUNet(
        in_channels=1,
        out_channels=1,
        img_size=(352, 192, 192),
        feature_size=4,
        hidden_size=384,
        mlp_dim=768,
        num_heads=4,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0
    )
    model_config = torch.load(model_path, map_location='cuda:' + GPU)
    model.load_state_dict(model_config['net'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer.load_state_dict(model_config['optimizer'])
    last_epoch = model_config['epoch']
    best_val_loss = model_config['best_val_loss']
    if best_val_loss > 1:
        best_val_loss = 0
    best_val_epoch = model_config['best_val_epoch']

    return model, optimizer, last_epoch, best_val_loss, best_val_epoch


def get_scale_class_map(label):
    # label 1x16x16x16 return 1x16x16x16
    batch_size = label.shape[0]
    l, w, h = label.shape[1], label.shape[2], label.shape[3]
    Z = torch.Tensor([]).float()
    for batch in range(batch_size):
        out = torch.round(torch.from_numpy(ndimage.distance_transform_edt(torch.Tensor.cpu(label[batch]))))[np.newaxis,
              :l, :w, :h].float()
        Z = torch.cat((Z, out), dim=0)
    return Z


# 生成 train.txt, val.txt 训练集和验证集的文件列表
class train_val_split():
    def __init__(self, rootpath, datapath, train_percent):
        self.rootpath = rootpath
        self.datapath = datapath
        self.train_percent = train_percent
        self.split_expand()

    # 包含增广图像的数据集切分，确保训练和验证集不会有相同图片信息
    def split_expand(self):
        print('split dataset')
        train_path, val_path = '../train.txt', \
                               '../val.txt'
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
            '61.nii.gz', '62.nii.gz', '63.nii.gz', '64.nii.gz', '65.nii.gz', '66.nii.gz', '67.nii.gz', '68.nii.gz',
            '69.nii.gz', '70.nii.gz', '71.nii.gz', '72.nii.gz', '73.nii.gz', '74.nii.gz', '75.nii.gz',
            '76.nii.gz', '77.nii.gz', '78.nii.gz', '79.nii.gz', '80.nii.gz', '81.nii.gz', '82.nii.gz', '83.nii.gz',
            '84.nii.gz', '85.nii.gz', '86.nii.gz', '87.nii.gz'
        ]
        data_list = ['01.nii.gz', '02.nii.gz', '03.nii.gz', '04.nii.gz', '05.nii.gz', '06.nii.gz', '07.nii.gz',
                     '08.nii.gz', '09.nii.gz', '10.nii.gz', '11.nii.gz', '12.nii.gz', '13.nii.gz', '14.nii.gz',
                     '15.nii.gz',
                     '16.nii.gz', '17.nii.gz', '18.nii.gz', '19.nii.gz', '20.nii.gz', '21.nii.gz', '22.nii.gz',
                     '23.nii.gz', '24.nii.gz', '25.nii.gz', '26.nii.gz', '27.nii.gz', '28.nii.gz', '29.nii.gz',
                     '30.nii.gz',
                     '31.nii.gz', '32.nii.gz', '33.nii.gz', '34.nii.gz', '35.nii.gz', '36.nii.gz', '37.nii.gz',
                     '38.nii.gz', '39.nii.gz', '40.nii.gz', '41.nii.gz', '42.nii.gz', '43.nii.gz', '44.nii.gz',
                     '45.nii.gz',
                     '46.nii.gz', '47.nii.gz', '48.nii.gz', '49.nii.gz', '50.nii.gz', '51.nii.gz', '52.nii.gz',
                     '53.nii.gz', '54.nii.gz', '55.nii.gz', '56.nii.gz', '57.nii.gz', '58.nii.gz', '59.nii.gz',
                     '60.nii.gz',
                     '61.nii.gz', '62.nii.gz', '63.nii.gz', '64.nii.gz', '65.nii.gz', '66.nii.gz', '67.nii.gz',
                     '68.nii.gz', '69.nii.gz', '70.nii.gz', '71.nii.gz', '72.nii.gz', '73.nii.gz', '74.nii.gz',
                     '75.nii.gz',
                     '76.nii.gz', '77.nii.gz', '78.nii.gz', '79.nii.gz', '80.nii.gz', '81.nii.gz', '82.nii.gz',
                     '83.nii.gz', '84.nii.gz', '85.nii.gz', '86.nii.gz', '87.nii.gz']  # 原文件列表

        print('augmentation is ' + str(args.augmentation))
        if args.augmentation:
            disk_data_list = os.listdir('../../images/')  # 实际文件列表
        else:
            disk_data_list = os.listdir('../../images87/')  # 实际文件列表
        list_len = 87
        pos_len = 52
        neg_len = 35
        train_len = self.train_percent * list_len
        pos_train_len = self.train_percent * pos_len
        neg_train_len = self.train_percent * neg_len
        shuffle_list = list(range(list_len))
        pos_shuffle_list = list(range(pos_len))
        neg_shuffle_list = list(range(neg_len))
        random.shuffle(shuffle_list)
        random.shuffle(pos_shuffle_list)
        random.shuffle(neg_shuffle_list)
        train_list, val_list = [], []
        full_train_list, full_val_list = [], []
        for i in range(pos_len):
            if i < pos_train_len:
                train_list.append(pos_data_list[pos_shuffle_list[i]])
                full_train_list.append(pos_data_list[pos_shuffle_list[i]])
            else:
                val_list.append(pos_data_list[pos_shuffle_list[i]])
                full_val_list.append(pos_data_list[pos_shuffle_list[i]])

        for i in range(neg_len):
            if i < neg_train_len:
                train_list.append(neg_data_list[neg_shuffle_list[i]])
                full_train_list.append(neg_data_list[neg_shuffle_list[i]])
            else:
                val_list.append(neg_data_list[neg_shuffle_list[i]])
                full_val_list.append(neg_data_list[neg_shuffle_list[i]])

        # 将增广图片加入训练和验证集
        for idx, (train_file) in enumerate(train_list):
            print('training set splitting progress: ' + str(round(idx / len(train_list) * 100, 2)) + '%')
            train_file_num = int(train_file.split('.')[0])
            for disk_file in disk_data_list:
                disk_file_num = int(disk_file.split('.')[0])
                if not train_file_num == disk_file_num:
                    if abs(train_file_num - disk_file_num) % list_len == 0:
                        full_train_list.append(disk_file)

        for idx, (val_file) in enumerate(val_list):
            print('validating set splitting progress: ' + str(round(idx / len(val_list) * 100, 2)) + '%')
            val_file_num = int(val_file.split('.')[0])
            for disk_file in disk_data_list:
                disk_file_num = int(disk_file.split('.')[0])
                if not val_file_num == disk_file_num:
                    if abs(val_file_num - disk_file_num) % list_len == 0:
                        full_val_list.append(disk_file)

        print('training and validating set length is ' + str(len(full_train_list)) + ' : ' + str(len(full_val_list)))

        with open(train_path, 'w') as f:
            for train_name in full_train_list:
                f.write(train_name + '\n')
        with open(val_path, 'w') as f:
            for val_name in full_val_list:
                f.write(val_name + '\n')


def inspect_nan(ts):
    # 查看tensor中是否出现nan
    return ts.isnan().any().item()


def sum_nan(ts):
    # 查看tensor中nan的个数
    nan_num = torch.sum(torch.where(ts.isnan(), 1, 0)).item()
    return nan_num


def inspect_inf(ts):
    # 查看tensor中是否出现inf
    return ts.isinf().any().item()


def sum_inf(ts):
    # 查看tensor中inf的个数
    inf_num = torch.sum(torch.where(ts.isinf(), 1, 0)).item()
    return inf_num


# 获取伪骨架
def gen_pseudo_skeleton(output_P):
    return torch.where(output_P > args.T_p, 1, 0)


# 获取分割mask
def gen_segmentation_map(output_P):
    return torch.where(output_P > 0.5, 1, 0)


def get_3Dshape(ts):
    return ts.shape[0], ts.shape[1], ts.shape[2]


# 获取mask上点的坐标
def get_skeleton_u(skeleton):
    l, w, h = get_3Dshape(skeleton)
    ls = []
    for i in range(l):
        for j in range(w):
            for k in range(h):
                if skeleton[i][j][k] == 1:
                    ls.append([i, j, k])
    return ls


def Visulization(sample_name, fmap):
    fmap = fmap.detach().cpu().numpy()
    sample_num = sample_name.split('.')[0]
    save_path = '../vis/'
    c = fmap.shape[1]
    for f in range(c):
        ar = fmap[0, f, :, :, :]
        im = itk.GetImageFromArray(ar)
        itk.WriteImage(im, save_path + sample_num + '_' + str(f) + '.nii.gz')
