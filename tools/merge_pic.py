import os 
import cv2 as cv
import numpy as np


""" 从训练和测试文件夹中读取swan图像再拼接phase图像 """
stage2_pics_dir="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"

swan_train_dir="/mnt/hdd1/zhulu/hospital/images/train"
swan_val_dir="/mnt/hdd1/zhulu/hospital/images/val"
phase_file_name="Series-FILT_PHA_ Ax SWAN new"
# swan_train_imgs=os.listdir(swan_train_dir)
# swan_val_imgs=os.listdir(swan_val_dir)

def merge(swan_dir,stage2_pics_dir,split):
    if split=="train":
        output_dir="/mnt/hdd1/zhulu/hospital/fuse_images/train"
    else:
        output_dir="/mnt/hdd1/zhulu/hospital/fuse_images/val"
    
    swan_imgs=os.listdir(swan_dir)
    for swan_img in swan_imgs:
        swan_img_path=os.path.join(swan_dir,swan_img)
        swan_image=cv.imread(swan_img_path,cv.IMREAD_GRAYSCALE)  #转换为灰度图像
        
        #利用病人信息去找文件夹，利用图层信息去找对应的图像
        patient_id=swan_img.split("_")[0]
        img_layer=swan_img.split("_")[1].split("-")[-1]  #自带一个png后缀了
        phase_dir=os.path.join(stage2_pics_dir,patient_id,phase_file_name)
        phase_imgs=os.listdir(phase_dir)
        phase_prefix="-".join(phase_imgs[0].split("-")[:2])
        
        phase_img_path=os.path.join(phase_dir,phase_prefix+"-"+img_layer)
        phase_img=cv.imread(phase_img_path,cv.IMREAD_GRAYSCALE)
        
        third_channel=np.zeros_like(swan_image)
        
        
        fuse_img=np.stack([swan_image,phase_img,third_channel],axis=-1)
        
        output_img_path = os.path.join(output_dir, swan_img)
        cv.imwrite(output_img_path, fuse_img)
        
        

        
        
merge(swan_train_dir,stage2_pics_dir,"train")    
merge(swan_val_dir,stage2_pics_dir,"val")    
        
        
        
        
    
    
    



