# Code for HMSAU-Net

> Automatic Detection Adenoid Hypertrophy in CBCT Based on Deep Learning

## Train HMSAU-Net with your own dataset
1. Unzip all files to your working directory.
2. Create a folder named output at your working directory.
3. Create a folder named images at your working directory, then put your own dataset into it.
4. Use Utils.train_val_split.split_expand to generate train.txt and val.txt that contains the training list and validating list.
5. Install python3.7 and all dependent packages in requirements.txt.
6. Change to Learn directory and execute command: "CUDA_VISIBLE_DEVICES=0 nohup python -u Train_saunet.py &".
7. Execute command: "tail -f nohup.out" for training detail inspection.

## Test HMSAU-Net
1. Execute command: "CUDA_VISIBLE_DEVICES=0 nohup python -u Test_saunet.py &".
2. Execute command: "tail -f nohup.out" for validating detail inspection.

![流程图github](https://user-images.githubusercontent.com/24643110/182013346-dab4796c-40fd-4399-8423-48f10defb032.png)
