from Data import MyDataSet_Crop
import numpy as np
from torch.utils.data import DataLoader
import Config
import SimpleITK as sitk
from Funcs import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))
args = Config.args
device = torch.device('cuda:0')


def test_segmentation(test_loader):
    # 获取训练好的模型
    model, optimizer, last_epoch = Utils.load_model_rev(device, '0')

    model.eval()  # evaluate 不启用BN和dropout

    savepath = os.path.join(args.rootpath, 'output')

    image = sitk.ReadImage('/data/lak/Label_new/001.nii')
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    with torch.no_grad():
        dice_sum = 0
        count = 0
        for data, label, sample_name in test_loader:
            data = data.float().to(device)
            label = label.float().to(device)
            output_P = model(data)
            pred = Utils.gen_segmentation_map(output_P)
            pred_np = np.array(torch.transpose(pred[0][0], 0, 2).cpu().numpy(), dtype='uint8')
            # pred_np = np.array(pred[0][0].cpu().numpy(), dtype='uint8')
            img = sitk.GetImageFromArray(pred_np)
            img.SetOrigin(origin)
            img.SetSpacing(spacing)
            img.SetDirection(direction)

            sitk.WriteImage(img, os.path.join(savepath, sample_name[0]))

            dice = calculate_dice_coefficient(pred, label)
            print('dice of [' + str(sample_name[0]) + '] is ' + str(dice.item()))
            dice_sum = dice_sum + dice.item()
            count = count + 1
        dice_sum = dice_sum / count
        print('average dice is ' + str(dice_sum) + '')
        print('complete predicting')
    return dice_sum


def test_segmentation_crop(test_loader):
    model, optimizer, last_epoch, best_val_loss, best_val_epoch = Utils.load_model_unet(device, '0',
                                                                                        os.path.join('..', 'Save',
                                                                                                     'best.pth'))

    model.eval()  # evaluate 不启用BN和dropout

    savepath = os.path.join(args.rootpath, 'output')

    image = sitk.ReadImage('/data1/lak/images/01.nii.gz')
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    with torch.no_grad():

        dice_sum = 0
        count = 0
        for ori_data, ori_label, sample_name in test_loader:
            ori_data = ori_data.to(torch.float).to(device)
            ori_label = ori_label.to(torch.long).to(device)
            out = model(ori_data)
            pred = Utils.gen_segmentation_map(out)
            pred_np = np.array(pred[0][0].cpu().numpy(), dtype='uint8')  # [:, ::-1, ::-1]
            # pred_np = np.array(pred[0][0].cpu().numpy(), dtype='uint8')
            img = sitk.GetImageFromArray(pred_np)
            img.SetOrigin(origin)
            img.SetSpacing(spacing)
            img.SetDirection(direction)

            sitk.WriteImage(img, os.path.join(savepath, sample_name[0]))

            dice = calculate_dice_coefficient(pred, ori_label.to(device))
            print('dice of [' + str(sample_name[0]) + '] is ' + str(dice.item()))
            dice_sum = dice_sum + dice.item()
            count = count + 1
        dice_sum = dice_sum / count
        print('average dice is ' + str(dice_sum) + '')
        print('complete predicting')
        return dice_sum


def Test_UNet():
    test_set = MyDataSet_Crop.MyDataset('../val.txt', stage='test')
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    test_segmentation_crop(test_loader)


if __name__ == '__main__':
    Test_UNet()
