import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
import torch
import torch.nn as nn
import Config
args = Config.args

class ops(nn.Module):  # two (Conv+BN+ReLU)
    def __init__(self, channel_in, channel_inter, channel_out, kernel, stride, padding):
        super(ops, self).__init__()
        self.ac, self.mp = nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2)
        self.layer_1 = nn.Conv3d(channel_in, channel_inter, kernel, stride, padding)
        self.bn_1 = nn.BatchNorm3d(channel_inter)
        self.layer_2 = nn.Conv3d(channel_inter, channel_out, kernel, stride, padding)
        self.bn_2 = nn.BatchNorm3d(channel_out)

    def forward(self, x):
        Layer_1 = self.layer_1(x)
        BN_1 = self.bn_1(Layer_1)
        Pool_1 = self.ac(BN_1)
        Layer_2 = self.layer_2(Pool_1)
        BN_2 = self.bn_2(Layer_2)
        Pool_2 = self.ac(BN_2)
        return Pool_2


class ops_last(nn.Module):  # 最后一次卷积加预测P和Zhat
    def __init__(self, channel_in, channel_inter, kernel, stride, padding):
        super(ops_last, self).__init__()
        self.layer = nn.Conv3d(channel_in, channel_inter, kernel, stride, padding)
        self.layer_binary = nn.Conv3d(1, 2, kernel, stride, padding)
        self.layer_K = nn.Conv3d(channel_in, args.K, kernel, stride, padding)
        self.ac1 = nn.Sigmoid()

    def forward(self, x):
        Layer = self.layer(x) # x:[1, 16, l, w, h]
        output_P = self.ac1(Layer)
        return output_P#, output_Z


class UNet3D(nn.Module):
    def __init__(self, filter_lists=[1, 8, 16, 32, 64, 128, 1], kernel=3, stride=1, padding=1): #[1, 16, 32, 64, 128, 256, 1]
        super(UNet3D, self).__init__()

        self.ac, self.mp = nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2)
        self.down_1 = ops(filter_lists[0], filter_lists[1], filter_lists[2], kernel, stride, padding)
        self.down_2 = ops(filter_lists[2], filter_lists[2], filter_lists[3], kernel, stride, padding)
        self.down_3 = ops(filter_lists[3], filter_lists[3], filter_lists[4], kernel, stride, padding)
        self.down_4 = ops(filter_lists[4], filter_lists[4], filter_lists[5], kernel, stride, padding)
        self.re_3 = nn.ConvTranspose3d(filter_lists[5], filter_lists[5], 2, stride=2)
        self.up_3 = ops(filter_lists[4] + filter_lists[5], filter_lists[4], filter_lists[4], kernel, stride, padding)
        self.re_2 = nn.ConvTranspose3d(filter_lists[4], filter_lists[4], 2, stride=2)
        self.up_2 = ops(filter_lists[3] + filter_lists[4], filter_lists[3], filter_lists[3], kernel, stride, padding)
        self.re_1 = nn.ConvTranspose3d(filter_lists[3], filter_lists[3], 2, stride=2)
        self.up_1 = ops(filter_lists[2] + filter_lists[3], filter_lists[2], filter_lists[2], kernel, stride, padding)
        self.output = ops_last(filter_lists[2], filter_lists[6], 1, 1, 0)

    def forward(self, mp_1):
        # Down
        Down_1 = self.down_1(mp_1)
        mp_2 = self.mp(Down_1)
        Down_2 = self.down_2(mp_2)
        mp_3 = self.mp(Down_2)
        Down_3 = self.down_3(mp_3)
        mp_4 = self.mp(Down_3)
        Down_4 = self.down_4(mp_4)
        # Up
        Re_3 = self.re_3(Down_4)
        Up_3 = self.up_3(torch.cat((Down_3, Re_3), dim=1))
        Re_2 = self.re_2(Up_3)
        Up_2 = self.up_2(torch.cat((Down_2, Re_2), dim=1))
        Re_1 = self.re_1(Up_2)
        Up_1 = self.up_1(torch.cat((Down_1, Re_1), dim=1))
        # output
        output_P = self.output(Up_1)  # 标签输出
        return output_P