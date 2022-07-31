# loss function以及一些计算函数
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import Utils
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
crop_size = [96, 96, 96]
crop_stride = [96, 96, 96]

class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # batch * loss
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, P, Z_hat, label, Z):
        P = nn.Sigmoid()(P)
        P = torch.squeeze(P, dim=1)  # (m, l, w, h)

        beta_p = 0.5 / torch.sum(label)
        beta_n = 0.5 / torch.sum(1 - label)
        # P = torch.clamp(P, min=1.0e-6, max=1.0 - (1.0e-6))  # 裁剪
        loss = - torch.sum(
            torch.mul(beta_p, torch.mul(label, torch.log(P))) +
            torch.mul(beta_n, torch.mul(1.0 - label, torch.log(1.0 - P)))
        )
        return loss

def calculate_dice_coefficient(pred, target):
    smooth = 1.
    num = pred.size(0)  # 获取样本量
    print(pred.shape)
    m1 = pred.view(num, -1)  # 按照样本量展平
    m2 = target.view(num, -1)  # 按照样本量展平
    intersection = (m1 * m2).sum()  # 交集元素数量
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return dice_sum / class_num

class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size()[1]

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num

class DiceMeanLoss_HM(nn.Module):
    def __init__(self):
        super(DiceMeanLoss_HM, self).__init__()
        self.mp = nn.MaxPool3d(2, 2)

    def cal(self, logits, targets):
        class_num = logits.size()[1]

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num

    def forward(self, out1, out2, out3, out4, out5, targets):
        loss1 = self.cal(out1, targets)
        loss2 = self.cal(out2, self.mp(targets))
        loss3 = self.cal(out3, self.mp(self.mp(targets)))
        loss4 = self.cal(out4, self.mp(self.mp(self.mp(targets))))
        loss5 = self.cal(out5, self.mp(self.mp(self.mp(self.mp(targets)))))
        return (loss1 + loss2 + loss3 + loss4 + loss5) / 5