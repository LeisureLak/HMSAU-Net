import os

# place the train.txt and val.txt in the following directory
file_path = '/data/lak/unet3-d-master/'

train_file_name = 'train.txt'
val_file_name = 'val.txt'
diag_train_file_name = 'train_diag.txt'
diag_val_file_name = 'val_diag.txt'
diag_path = '../../Masks/'
ori_file_num = 87

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
             '61.nii.gz', '62.nii.gz', '63.nii.gz', '64.nii.gz', '65.nii.gz', '66.nii.gz', '67.nii.gz', '68.nii.gz', '69.nii.gz', '70.nii.gz', '71.nii.gz', '72.nii.gz', '73.nii.gz', '74.nii.gz', '75.nii.gz',
             '76.nii.gz', '77.nii.gz', '78.nii.gz', '79.nii.gz', '80.nii.gz', '81.nii.gz', '82.nii.gz', '83.nii.gz', '84.nii.gz', '85.nii.gz', '86.nii.gz', '87.nii.gz'
]


if __name__ == '__main__':
    train_file_path = file_path + train_file_name
    val_file_path = file_path + val_file_name
    diag_train_file_path = file_path + diag_train_file_name
    diag_val_file_path = file_path + diag_val_file_name

    print('train transfer')
    with open(train_file_path, 'r') as f:
        name_list = f.readlines()
        for name in name_list:
            num = int(name.split('.')[0])
            print(str(num) + ' --> ', end='')
            num = num % ori_file_num
            if num == 0:
                num = ori_file_num
            print(str(num) + ' --> ', end='')
            find = False
            for p in pos_data_list:
                ori_num = int(p.split('.')[0])
                if num == ori_num:
                    label = 0
                    find = True
                    break;
            if find == False:
                label = 1
            print(str(label))
            diag_name = diag_path + str(label) + '/' + name
            if not os.path.exists(diag_train_file_path):
                with open(diag_train_file_path, 'w') as df:
                    df.writelines('')
                    df.close()
            with open(diag_train_file_path, 'a') as df:
                df.writelines(diag_name)
                df.close()
        f.close()

    print('val transfer')
    with open(val_file_path, 'r') as f:
        name_list = f.readlines()
        for name in name_list:
            num = int(name.split('.')[0])
            print(str(num) + ' --> ', end='')
            num = num % ori_file_num
            if num == 0:
                num = ori_file_num
            print(str(num) + ' --> ', end='')
            find = False
            for p in pos_data_list:
                ori_num = int(p.split('.')[0])
                if num == ori_num:
                    label = 0
                    find = True
                    break;
            if find == False:
                label = 1
            print(str(label))
            diag_name = diag_path + str(label) + '/' + name
            if not os.path.exists(diag_val_file_path):
                with open(diag_val_file_path, 'w') as df:
                    df.writelines('')
                    df.close()
            with open(diag_val_file_path, 'a') as df:
                df.writelines(diag_name)
                df.close()

        f.close()
