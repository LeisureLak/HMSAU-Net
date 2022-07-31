from torch.utils.tensorboard import SummaryWriter
import math
import Config
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Data import MyDataSet_Crop as MyDataSet_Crop
from Net import UNet3D as net
from Funcs import *
from torch.cuda.amp import autocast as autocast, GradScaler
from Utils import Utils
from Utils import Cal_Utils

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
args = Config.args


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


class train_model(nn.Module):
    def __init__(self, model, train_loader, val_loader):
        super(train_model, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    # 训练原始unet
    def train_crop(self, optimizer, epoch):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        i = 0
        average_loss = 0
        # 该循环结束走完所有的图片
        for dataset, labelset, sample_name in self.train_loader:  # (1, sample_num, 1, l, w, h)
            # 单张图片
            dataset, labelset = torch.squeeze(dataset, dim=0), torch.squeeze(labelset,
                                                                             dim=0)  # dataloader添加的维度去除 --> (sample_num, 1, l, w, h)
            dataset, labelset = dataset.to(torch.float32).to(device), labelset.to(torch.float32).to(
                device)  # 568 * 64 * 64 * 64
            # 根据batch_size将一张图的所有切块分成batch，进行训练，该循环结束之后是一张图的所有切块被训练完
            for j in range(dataset.shape[0] // args.batch_size_crop):
                start = j * args.batch_size_crop
                end = start + args.batch_size_crop
                data = dataset[start:end, :, :, :, :].to(device)
                label = labelset[start:end, :, :, :, :].to(device)

                optimizer.zero_grad()  # 梯度归零
                with autocast():
                    # Z = Utils.get_scale_class_map(label).to(device)
                    output_P = self.model(data)
                    # output_P = self.model(data)
                    # loss = WeightedCrossEntropyLoss()(output_P, None, label, None)
                    loss = DiceMeanLoss()(output_P, label)
                writer.add_scalar('loss of ' + str(sample_name), loss.item(), i)
                average_loss = average_loss + loss.item()
                if math.isnan(loss):
                    print('loss is nan!')  # 停止传播
                    # quit()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                #     optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
            print(
                'Training loss = ' + str(loss.item()) + ', (batch = ' + str(i) + ', epoch = ' + str(
                    epoch) + ', lr = ' + str(optimizer.state_dict()['param_groups'][0]['lr']) + '), Samples:' + str(
                    sample_name))
            i = i + 1
        # 除以所有图片的所有切块数量计算平均值
        average_loss = average_loss / i
        return average_loss

    def val_crop(self):  # TODO
        self.model.eval()  # evaluate 不启用BN和dropout

        with torch.no_grad():
            dice_sum = 0
            count = 0
            for ori_data, ori_label, sample_name in self.val_loader:
                ori_data = ori_data.to(torch.float).to(device)
                ori_label = ori_label.to(torch.long).to(device)
                out = self.model(ori_data)
                pred = Utils.gen_segmentation_map(out)
                dice = calculate_dice_coefficient(pred, ori_label.to(device))
                print('dice of [' + str(sample_name[0]) + '] is ' + str(dice.item()))
                dice_sum = dice_sum + dice.item()
                count = count + 1
            dice_sum = dice_sum / count
            print('average dice is ' + str(dice_sum) + '')
            print('complete predicting')
            return dice_sum


def train_unet():
    # 环境路径
    datapath = args.datapath
    labelpath = args.labelpath

    # Utils.train_val_split(rootpath, datapath, 0.8)

    # 存在旧模型
    if os.path.exists('../Save/last_model_unet.pth'):
        print('there is old model, try loading it...')
        model, optimizer, begin_epoch, best_val_loss, best_val_epoch = Utils.load_model_unet(device, '0',
                                                                                             os.path.join('..', 'Save',
                                                                                                          'last_model_unet.pth'))
    else:
        # Utils.train_val_split('../', datapath, 0.8)
        print('there is no old model, creating new model...')
        model = net.UNet3D()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr_crop)  # 0.0001 20
        begin_epoch = -1
        best_val_loss = 0
        best_val_epoch = -1

    # 通过文件列表初始化dataset
    train_set = MyDataSet_Crop.MyDataset('../train.txt', datapath, labelpath,
                                         batchsize=args.batch_size_crop)
    val_set = MyDataSet_Crop.MyDataset('../val.txt', datapath, labelpath,
                                       stage='test')

    train_loader = DataLoader(dataset=train_set, shuffle=True)
    print('trainset length: ', str(len(train_loader)))
    val_loader = DataLoader(dataset=val_set, batch_size=1)
    print('validatingset length: ', str(len(val_loader)))
    # model.train()  # 启用BN和dropout
    # 初始化训练模型
    train_it = train_model(model, train_loader, val_loader)

    epochs = args.num_epochs

    last_dice = 1
    for epoch in range(begin_epoch + 1, epochs):
        # Utils.adjust_learning_rate(optimizer, epoch)
        print('training loss......, (epoch = ' + str(epoch) + ', best_epoch = ' + str(
            best_val_epoch) + ', best_dice = ' + str(best_val_loss) + ')')
        average_loss = train_it.train_crop(optimizer, epoch)
        dice = train_it.val_crop()

        if dice > best_val_loss:
            best_val_loss = dice
            best_val_epoch = epoch
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'best_val_loss': best_val_loss, 'best_val_epoch': best_val_epoch}
            torch.save(state, os.path.join('..', 'Save', 'best.pth'))
        elif dice < last_dice:
            Utils.immediately_adjust_learning_rate(optimizer)
        last_dice = dice
        writer.add_scalar('val_loss', dice, epoch)
        writer.add_scalar('average_loss', average_loss, epoch)
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'best_val_loss': best_val_loss, 'best_val_epoch': best_val_epoch}
        torch.save(state, os.path.join('..', 'Save', 'last_model_unet.pth'))
        torch.cuda.empty_cache()
    writer.close()
    print('complete training.')


if __name__ == '__main__':
    if args.hp:
        scaler = GradScaler()
    device = torch.device('cuda:0')
    writer = SummaryWriter('logs')
    train_unet()
