import SimpleITK as sitk
import numpy as np
import os
import cv2
def Pre_Class(name,img_np,label_np,indate):

    img_dir = "/home/ubuntu/liuyiyao/Data/Breast_w_class/img"
    mask_dir = "/home/ubuntu/liuyiyao/Data/Breast_w_class/label"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(mask_dir):
        os.makedirs(mask_dir)
    new_label = np.zeros_like(label_np)
    new = int(indate)+1
    new_label[label_np>=1] = new
    print(int(indate)+1)
    img = sitk.GetImageFromArray(img_np)
    label = sitk.GetImageFromArray(new_label)
    sitk.WriteImage(img,img_dir+"/"+name)
    sitk.WriteImage(label,mask_dir+"/"+name)



img_root = '/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/img'
label_root = '/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/label'
label1_root="/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/great_data_large"
label_names = os.listdir(label_root)
j=0
for name in label_names:

    label_data = sitk.ReadImage(label_root + '/' + name)
    label_np = sitk.GetArrayFromImage(label_data)
    img_data = sitk.ReadImage(img_root + '/' + name)
    img_np = sitk.GetArrayFromImage(img_data)
    print(name.split('.')[0])
    source = open(label1_root + '/' + name.split('.')[0] + '/label.txt')  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    Pre_Class(name,img_np,label_np,indate)







    j+=1

