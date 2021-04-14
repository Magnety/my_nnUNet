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

def Slicer(name,img_np,label_np,indate):

    img_dir = "/home/ubuntu/liuyiyao/Data/Breast_slicer/img"
    mask_dir = "/home/ubuntu/liuyiyao/Data/Breast_slicer/label"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(mask_dir):
        os.makedirs(mask_dir)
    a = []

    for i in range(label_np.shape[0]):
        (mean, stddv) = cv2.meanStdDev(label_np[i, :, :])
        if mean > 0:
            a.append(i)
    a_s = a[0]
    a_e = a[len(a) - 1]
    if a_s > 30:
        a_s -= 30
    else:
        a_s = 0
    if a_e < 288:
        a_e += 30
    else:
        a_e = 318
    img = sitk.GetImageFromArray(img_np[a_s:a_e,:,:])
    label = sitk.GetImageFromArray(label_np[a_s:a_e,:,:])
    sitk.WriteImage(img,img_dir+"/"+name)
    sitk.WriteImage(label,mask_dir+"/"+name)
def Slicer_match(name,img_np,label_np,indate):

    img_dir = "/home/ubuntu/liuyiyao/Data/Breast_match/img"
    mask_dir = "/home/ubuntu/liuyiyao/Data/Breast_match/label"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(mask_dir):
        os.makedirs(mask_dir)
    a = []
    for i in range(label_np.shape[0]):
        (mean, stddv) = cv2.meanStdDev(label_np[i, :, :])
        if mean > 0:
            a.append(i)
    a_s = a[0]
    a_e = a[len(a) - 1]
    if a_s > 5:
        a_s -= 5
    else:
        a_s = 0
    if a_e < 313:
        a_e += 5
    else:
        a_e = 318
    img = sitk.GetImageFromArray(img_np[a_s:a_e,:,:])
    label = sitk.GetImageFromArray(label_np[a_s:a_e,:,:])
    sitk.WriteImage(img,img_dir+"/"+name)
    sitk.WriteImage(label,mask_dir+"/"+name)



img_root = '/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/img'
label_root = '/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/label'
label1_root="/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/great_data_large"
label_names = os.listdir(label_root)
j=0
i=0
k=0
for name in label_names:

    label_data = sitk.ReadImage(label_root + '/' + name)
    label_np = sitk.GetArrayFromImage(label_data)
    img_data = sitk.ReadImage(img_root + '/' + name)
    img_np = sitk.GetArrayFromImage(img_data)
    print(name.split('.')[0])
    source = open(label1_root + '/' + name.split('.')[0] + '/label.txt')  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    #Slicer(name,img_np,label_np,indate)
    if int(indate)==1:
        i+=1
    else:
        k+=1
    j+=1
print(i,k,j)

