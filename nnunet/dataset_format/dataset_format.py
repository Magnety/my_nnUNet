import SimpleITK as sitk
import numpy as np
import os
img_path = "/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/img"
mask_path = "/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/label"
label_path = "/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/great_data_large"


mask_names = os.listdir(mask_path)
j=0
for name in mask_names:
    dir = "/home/ubuntu/liuyiyao/Data/Breast/case_%05.0d"%j
    if not os.path.isdir(dir):
        os.makedirs(dir)
    mask_data = sitk.ReadImage(mask_path + '/' + name)
    img_data = sitk.ReadImage(img_path + '/' + name)
    sitk.WriteImage(img_data,dir+'/imaging.nii.gz')
    sitk.WriteImage(mask_data,dir+'/segmentation.nii.gz')
    source = open(label_path + '/' + name.split('.')[0] + '/label.txt')  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    target = open(dir + '/label.txt', 'w')  # 打开目的文件
    target.write(indate)
    j+=1
