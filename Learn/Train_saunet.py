import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
import math
import torch
from torch.cuda.amp import autocast as autocast, GradScaler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import Config
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Data import MyDataSet as MyDataSet
from Net import SAUNet
from Funcs import *

args = Config.args

class train_model(nn.Module):
    def __init__(self, model, train_loader, val_loader):
        super(train_model, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, optimizer, epoch):
        torch.autograd.set_detect_anomaly(True)

        i = 0
        average_loss = 0
        loss_fn = DiceMeanLoss()
        for data, label, sample_name in self.train_loader:
            data, label = data.to(torch.float16).to(device), label.to(torch.float16).to(device)

            optimizer.zero_grad()

            with autocast():
                output_P = self.model(data)
                loss = loss_fn(output_P, label)
            writer.add_scalar('loss of ' + str(sample_name), loss.item(), i)
            average_loss = average_loss + loss.item()
            if math.isnan(loss):
                print('loss is nan!')
            if args.hp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            print(
                'Training loss = ' + str(loss.item()) + ', (batch = ' + str(i) + ', epoch = ' + str(
                    epoch) + ', lr = ' + str(optimizer.state_dict()['param_groups'][0]['lr']) + '), Samples:' + str(
                    sample_name))
            i = i + 1
        average_loss = average_loss / i
        return average_loss

    def val(self):
        self.model.eval()

        with torch.no_grad():
            dice_sum = 0
            count = 0
            for ori_data, ori_label, sample_name in self.val_loader:
                ori_data = ori_data.to(torch.float).to(device)
                ori_label = ori_label.to(torch.long).to(device)
                out, _, _, _, _ = self.model(ori_data)
                pred = Utils.gen_segmentation_map(out)
                dice = calculate_dice_coefficient(pred, ori_label.to(device))
                # use with HM
                # dice = DiceMeanLoss_HM(pred, ori_label.to(device))
                print('dice of [' + str(sample_name[0]) + '] is ' + str(dice.item()))
                dice_sum = dice_sum + dice.item()
                count = count + 1
            dice_sum = dice_sum / count
            print('average dice is ' + str(dice_sum) + '')
            print('complete predicting')
            return dice_sum


def train_saunet():

    if os.path.exists('../Save/last_model_unet.pth'):
        print('there is old model, try loading it...')
        model, optimizer, begin_epoch, best_val_loss, best_val_epoch = Utils.load_model_saunet(device, '0', os.path.join('..', 'Save', 'last_model_unet.pth'))
    else:
        # Utils.train_val_split(rootpath, datapath, 0.8)
        print('there is no old model, creating new model...')
        model = saunet.SAUNet(
            in_channels=1,
            out_channels=1,
            img_size=(352, 192, 192),
            feature_size=4,
            hidden_size=96,
            mlp_dim=192,
            num_heads=4,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 0.0005 lr_crop
        # optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
        begin_epoch = -1
        best_val_loss = 0
        best_val_epoch = -1

    train_set = MyDataSet.MyDataset('../train.txt', stage='train')
    val_set = MyDataSet.MyDataset('../val.txt', stage='test')
    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1)

    train_it = train_model(model, train_loader, val_loader)

    epochs = args.num_epochs

    for epoch in range(begin_epoch + 1, epochs):
        print('training loss......, (epoch = ' + str(epoch) + ', best_epoch = ' + str(
            best_val_epoch) + ', best_dice = ' + str(best_val_loss) + ')')
        average_loss = train_it.train(optimizer, epoch)
        dice = train_it.val()

        if dice > best_val_loss:
            best_val_loss = dice
            best_val_epoch = epoch
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'best_val_loss': best_val_loss, 'best_val_epoch': best_val_epoch}
            torch.save(state, os.path.join('..', 'Save', 'best.pth'))
        else:
            Utils.immediately_adjust_learning_rate(optimizer)  # 一旦精度下降就减半lr
        writer.add_scalar('val_loss', dice, epoch)
        writer.add_scalar('average_loss', average_loss, epoch)
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'best_val_loss': best_val_loss, 'best_val_epoch': best_val_epoch}
        torch.save(state, os.path.join('..', 'Save', 'last_model_unet.pth'))
        torch.cuda.empty_cache()
    writer.close()
    print('complete training.')


if __name__ == '__main__':
    args.hp = True
    args.augmentation = True
    if args.hp:
        scaler = GradScaler()
    device = torch.device('cuda:0')
    writer = SummaryWriter('logs')
    train_saunet()
