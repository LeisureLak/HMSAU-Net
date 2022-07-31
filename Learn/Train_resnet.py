import os

from torch.cuda.amp import autocast as autocast, GradScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
from Net import Resnet
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import torch.nn as nn
import Config
import torch.optim as optim
from Data import MyDataSet_Diag as MyDataSet_Mask
from Data import MyDataLoader_Diag as MyDataLoader_Mask
from Utils import Utils
args = Config.args

def train_diagnosis():
    layers = 50

    if os.path.exists('../Save/last_model_resnet.pth'):
        print('there is old model, try loading it...')
        model, optimizer, begin_epoch, best_val_acc, best_val_epoch = Utils.load_model_resnet(device, '0', layers, os.path.join('..', 'Save', 'last_model_resnet.pth'))
    else:
        # Utils.train_val_split_diag(rootpath=os.path.join('..', 'Masks'), train_percent=0.8)
        print('there is no old model, creating new model...')
        model = resnet.generate_model(layers, n_classes=1, n_input_channels=2).to(device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 0.0001 20
        begin_epoch = -1
        best_val_acc = 0
        best_val_epoch = -1

    train_set = MyDataSet_Mask.MyDataset(rootpath='../', split_file_name='train_diag.txt', resize=False)
    val_set = MyDataSet_Mask.MyDataset(rootpath='../', split_file_name='val_diag.txt', resize=False, stage='train')

    train_loader = MyDataLoader_Mask.MyDataLoader(dataset=train_set, shuffle=True, batch_size=4)
    val_loader = MyDataLoader_Mask.MyDataLoader(dataset=val_set)

    loss_fn = nn.BCEWithLogitsLoss()
    epochs = 100
    last_acc = 0

    for epoch in range(begin_epoch + 1, epochs):
        # Utils.adjust_learning_rate(optimizer, epoch, lr = 0.001)
        i = 0

        # training
        model.train()
        average_loss = 0
        print('begin training...')
        for data, label, file_name in train_loader:
            print('Diagnosis resnet' + str(layers) + ' training loss......, (epoch = ' + str(epoch) + ', best_epoch = ' + str(
                best_val_epoch) + ', best_val_acc = ' + str(best_val_acc) + ')')
            data = data.to(device)
            label = torch.unsqueeze(label, dim=1).to(torch.float32).to(device)
            print(file_name)
            optimizer.zero_grad()

            if args.hp:
                with autocast():
                    out = model(data)
                    train_loss = loss_fn(out, label)
            else:
                out = model(data)
                train_loss = loss_fn(out, label)
            print('Diagnosis training loss = ' + str(train_loss.item()) + ', (batch = ' + str(i) + ', epoch = ' + str(
                    epoch) + ', lr = ' + str(optimizer.state_dict()['param_groups'][0]['lr']) + ')')
            average_loss += train_loss.item()
            if args.hp:
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                train_loss.backward()
                optimizer.step()
            i += 1
        average_loss = average_loss / i
        print('average training loss is : ' + str(average_loss))

        # validating
        model.eval()
        acc = 0
        print('begin validating...')
        right = 0
        with torch.no_grad():
            for data, label, file_name in val_loader:
                data = data.to(device)
                label = label.to(device)
                print(file_name)
                T = 0.5
                out = model(data)
                print('raw_out')
                print(out.view(-1))
                out = out.view(-1)
                out = nn.Sigmoid()(out)
                pred = torch.where(out > T, 1, 0)
                print('label')
                print(label.view(-1))
                print('out')
                print(out.view(-1))
                print('pred')
                print(pred.view(-1))
                # 预测正确的数量
                right += torch.sum(pred == label)
            acc = right / len(val_loader.dataset)
            print('validating accuracy is : ' + str(acc))

        # update best acc
        if acc > best_val_acc:
            best_val_acc = acc
            best_val_epoch = epoch
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'best_val_acc': best_val_acc, 'best_val_epoch': best_val_epoch}
            torch.save(state, os.path.join('..', 'Save', 'best_mask.pth'))
        elif acc < last_acc:
            Utils.immediately_adjust_learning_rate(optimizer)
        last_acc = acc

        writer.add_scalar('val_acc_mask', acc, epoch)
        writer.add_scalar('average_loss_mask', average_loss, epoch)

        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'best_val_acc': best_val_acc, 'best_val_epoch': best_val_epoch}
        torch.save(state, os.path.join('..', 'Save', 'last_model_resnet.pth'))
        torch.cuda.empty_cache()
    writer.close()
    print('complete training mask.')



if __name__ == '__main__':
    args.hp = True
    if args.hp:
        scaler = GradScaler()
    device = torch.device('cuda:0')
    writer = SummaryWriter('logs')
    train_diagnosis()
