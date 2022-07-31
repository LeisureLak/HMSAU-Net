import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
from Data import MyDataSet_Diag as MyDataSet_Mask
from Data import MyDataLoader_Diag as MyDataLoader_Mask
import numpy as np
from Utils import Cal_Utils
from Funcs import *
import Config

args = Config.args

device = torch.device('cuda:0')


def test_segmentation_diag_multimodel(test_loader):
    # 获取训练好的模型
    model_10, optimizer, last_epoch, best_val_loss, best_val_epoch = Utils.load_model_resnet(device, '0', os.path.join('..', 'Save', 'best_resnet' + str(10) + '.pth'))
    model_18, optimizer, last_epoch, best_val_loss, best_val_epoch = Utils.load_model_resnet(device, '0', os.path.join('..', 'Save', 'best_resnet' + str(18) + '.pth'))
    model_34, optimizer, last_epoch, best_val_loss, best_val_epoch = Utils.load_model_resnet(device, '0', os.path.join('..', 'Save', 'best_resnet' + str(34) + '.pth'))

    model_10.eval()
    model_18.eval()
    model_34.eval()
    with torch.no_grad():
        # validating
        print('begin testing...')
        right_10 = 0
        right_18 = 0
        right_34 = 0
        output_10 = []
        output_18 = []
        output_34 = []
        labels = []
        preds_10 = []
        preds_18 = []
        preds_34 = []
        file_names = []
        with torch.no_grad():
            for data, label, file_name in test_loader:
                data = data.to(device)
                label = label.to(device)
                file_names.extend(list(file_name))
                print('file ' + str(file_name) + ' is classified as ', end='')
                T = 0.5
                out_10 = model_10(data)
                out_18 = model_18(data)
                out_34 = model_34(data)
                out_10 = out_10.view(-1) # batch, p
                out_18 = out_18.view(-1) # batch, p
                out_34 = out_34.view(-1) # batch, p
                out_10 = nn.Sigmoid()(out_10)
                out_18 = nn.Sigmoid()(out_18)
                out_34 = nn.Sigmoid()(out_34)
                output_10.append(out_10)
                output_18.append(out_18)
                output_34.append(out_34)
                labels.append(label)
                pred_10 = torch.where(out_10 > T, 1, 0)
                pred_18 = torch.where(out_18 > T, 1, 0)
                pred_34 = torch.where(out_34 > T, 1, 0)
                print(str(pred_10.tolist()) + ', ' + str(pred_18.tolist()) + ', ' + str(pred_34.tolist()) + ' seperately by [resnet10,18,34] the Ground Truth is ' + str(label.tolist()))
                preds_10.append(pred_10)
                preds_18.append(pred_18)
                preds_34.append(pred_34)
                # 预测正确的数量
                right_10 += torch.sum(pred_10 == label)
                right_18 += torch.sum(pred_18 == label)
                right_34 += torch.sum(pred_34 == label)
            acc_10 = right_10 / len(test_loader.dataset)
            acc_18 = right_18 / len(test_loader.dataset)
            acc_34 = right_34 / len(test_loader.dataset)
            print('file_names: ' + str(file_names))
            print('validating accuracy of resnet10 is : ' + str(acc_10.item()) + ', ' + str(right_10.item()) + ' / ' + str(len(test_loader.dataset)) + '')
            print('validating accuracy of resnet18 is : ' + str(acc_18.item()) + ', ' + str(right_18.item()) + ' / ' + str(len(test_loader.dataset)) + '')
            print('validating accuracy of resnet34 is : ' + str(acc_34.item()) + ', ' + str(right_34.item()) + ' / ' + str(len(test_loader.dataset)) + '')

        # 转为二分类分数
        output_10 = torch.cat(output_10)
        output_2cls_10 = torch.unsqueeze(output_10, dim=1).cpu().numpy()
        score_10 = np.zeros((output_2cls_10.shape[0], 2))
        score_10[:, 1] = output_2cls_10[:, 0]
        score_10[:, 0] = 1 - output_2cls_10[:, 0]

        output_18 = torch.cat(output_18)
        output_2cls_18 = torch.unsqueeze(output_18, dim=1).cpu().numpy()
        score_18 = np.zeros((output_2cls_18.shape[0], 2))
        score_18[:, 1] = output_2cls_18[:, 0]
        score_18[:, 0] = 1 - output_2cls_18[:, 0]

        output_34 = torch.cat(output_34)
        output_2cls_34 = torch.unsqueeze(output_34, dim=1).cpu().numpy()
        score_34 = np.zeros((output_2cls_34.shape[0], 2))
        score_34[:, 1] = output_2cls_34[:, 0]
        score_34[:, 0] = 1 - output_2cls_34[:, 0]

        # label转one-hot
        print('file_names')
        print(file_names)
        print('Ground Truth labels:')
        print(torch.cat(labels))
        preds_10 = torch.cat(preds_10)
        preds_18 = torch.cat(preds_18)
        preds_34 = torch.cat(preds_34)

        labels = torch.cat(labels)



        Cal_Utils.Get_ROC_Curve_multimodel([output_10.cpu(), output_18.cpu(), output_34.cpu()], [labels.cpu(), labels.cpu(), labels.cpu()])

        print('resnet10指标')
        Cal_Utils.Cal_TP_FP_TN_FN_Sen_Spec_PPV_NPV_F1(labels.cpu(), preds_10.cpu())
        Cal_Utils.Get_Confusion_Matrix(labels.cpu(), preds_10.cpu(), 'resnet10')
        print('resnet18指标')
        Cal_Utils.Cal_TP_FP_TN_FN_Sen_Spec_PPV_NPV_F1(labels.cpu(), preds_18.cpu())
        Cal_Utils.Get_Confusion_Matrix(labels.cpu(), preds_18.cpu(), 'resnet18')
        print('resnet34指标')
        Cal_Utils.Cal_TP_FP_TN_FN_Sen_Spec_PPV_NPV_F1(labels.cpu(), preds_34.cpu())
        Cal_Utils.Get_Confusion_Matrix(labels.cpu(), preds_34.cpu(), 'resnet34')


def Test_Diag():

    test_set = MyDataSet_Mask.MyDataset(rootpath='../../output/', split_file_name='val_diag.txt', resize=False, stage='test', direct_get=True)
    test_loader = MyDataLoader_Mask.MyDataLoader(dataset=test_set, batch_size=1)

    test_segmentation_diag_multimodel(test_loader)


if __name__ == '__main__':
    Test_Diag()