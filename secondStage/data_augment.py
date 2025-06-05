from utils import rotate,flip,gaussianBlur
import os
import nibabel as nib

#将原本训练集中判断错误的进行数据增强重新训练，原本训练集就是nii数据

suffix_list=["_flip_horizontal.nii","_flip_vertical.nii","_rotae_90.nii","_rotate_180.nii","_rotate_270.nii","_gaussian_blur.nii"]
# 训练数据时的nii数据，数据包含磁敏感和相位图，16x32x16，旋转的时候需要切割一下
data_dir="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all" 

add_dir="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add2"


# #进行数据的删除
# delete_count=0
# for type_dir in os.listdir(data_dir):
#     datas=os.path.join(data_dir,type_dir)
#     for file_name in os.listdir(datas):
#         if any(suffix in file_name for suffix in suffix_list):
#             delete_count+=1
#             os.remove(os.path.join(datas, file_name))

wrong_pred_file="secondStage/phase_pred/test_wrong_predict_samples.txt"
with open(wrong_pred_file, 'r') as f:
    wrong_pred_samples = f.readlines()
    wrong_pred_samples = [x.strip() for x in wrong_pred_samples]
    
for i in range(len(wrong_pred_samples)):
    wrong_pred_sample=os.path.join(data_dir,wrong_pred_samples[i])
    img = nib.load(wrong_pred_sample)
    #翻转操作
    flip_horizontal=flip(img,1)
    flip_vertical=flip(img,0)
    flip_horizontal_name=wrong_pred_samples[i].split(".")[0]+"_flip_horizontal.nii"
    flip_vertical_name=wrong_pred_samples[i].split(".")[0]+"_flip_vertical.nii"
    nib.save(flip_horizontal,os.path.join(add_dir,flip_horizontal_name))
    nib.save(flip_vertical,os.path.join(add_dir,flip_vertical_name))
    
    #旋转操作
    rotate_90=rotate(img,90)   #顺时针旋转没问题，但是在ITK-SNAP中显示存在镜像问题
    rotate_180=rotate(img,180) #旋转部分都没有问题
    rotate_270=rotate(img,270)
    rotae_90_name=wrong_pred_samples[i].split(".")[0]+"_rotae_90.nii"
    rotate_180_name=wrong_pred_samples[i].split(".")[0]+"_rotate_180.nii"
    rotate_270_name=wrong_pred_samples[i].split(".")[0]+"_rotate_270.nii"
    nib.save(rotate_90,os.path.join(add_dir,rotae_90_name))
    nib.save(rotate_180,os.path.join(add_dir,rotate_180_name))
    nib.save(rotate_270,os.path.join(add_dir,rotate_270_name))
    
    
    #高斯滤波
    k_size=3
    gaussian_blur=gaussianBlur(img,k_size)
    gaussian_blur_name=wrong_pred_samples[i].split(".")[0]+"_gaussian_blur.nii"
    nib.save(gaussian_blur,os.path.join(add_dir,gaussian_blur_name))
    
    p=1+1
    