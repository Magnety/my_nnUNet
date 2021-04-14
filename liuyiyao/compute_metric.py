import SimpleITK as sitk
import numpy as np
from liuyiyao.eva import dc,jc,precision,recall,hd95,assd


gt_dir  = "/home/ubuntu/liuyiyao/my_nnUNet_data_raw_base/nnUNet_preprocessed/Task004_Breast_s/gt_segmentations"
predict_dir = "/home/ubuntu/liuyiyao/my_nnUNet_data_raw_base/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Breast_s/nnUNetTrainerV2__nnUNetPlansv2.1/all/validation_raw_postprocessed"
i=0
k=0
dice_all=0
for j in range(0,103):
    gt = sitk.ReadImage(gt_dir+"/case_%05.0d.nii.gz"%j)
    predict = sitk.ReadImage(predict_dir+"/case_%05.0d.nii.gz"%j)
    gt_np = sitk.GetArrayFromImage(gt)
    predict_np = sitk.GetArrayFromImage(predict)
    print("case_%05.0d"%j)
    dice = dc(predict_np,gt_np)
    if dice>0:
        i+=1
        dice_all+=dice
    else:
        k+=1

    print(dice)
dice_avg = dice_all/i
print("i:",i,"   k:",k,"   dice_avg:",dice_avg)

