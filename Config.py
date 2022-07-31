# 配置页面
import argparse
import sys
import os

sep = os.path.sep

sys.path.append(os.path.dirname(os.path.realpath(sys.argv[0])))

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--batch_size', default=2, type=int,
                    help='batch size for training')
parser.add_argument('--batch_size_crop', default=16, type=int,
                    help='batch size for crop training')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr_crop', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--split_rate', type=float, default=0.8,
                    help='rate of splitting the train and val data')
parser.add_argument('--channel_base', type=int, default=8,
                    help='basic number of the net\'s channel list')
parser.add_argument('--K', type=int, default=30,
                    help='the range of tubular structure scale classes（abandoned）')
parser.add_argument('--src_path', type=str, default='Unet3D',
                    help='relative path of the src root')
parser.add_argument('--src_path_crop', type=str, default='Unet3D',
                    help='relative path of the src root')
parser.add_argument('--hp', type=str, default=False,
                    help='use half precision')
parser.add_argument('--augmentation', type=str, default=True,
                    help='use half precision')


if (sys.platform == 'linux'):
    parser.add_argument('--rootpath', type=str, default='/data1/lak')
    parser.add_argument('--datapath', type=str, default='/data1/lak/images87')
    parser.add_argument('--labelpath', type=str, default='/data1/lak/segments87')
elif (sys.platform == 'win32'):
    parser.add_argument('--rootpath', type=str, default='D:/Tubular Structure')
    parser.add_argument('--datapath', type=str, default='D:/Tubular Structure/128x64x64/Image')
    parser.add_argument('--labelpath', type=str, default='D:/Tubular Structure/128x64x64/Label')

args = parser.parse_args()
