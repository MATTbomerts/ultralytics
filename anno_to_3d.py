import os
import json
from my_tool.utils import get_patient_data,sort_files,img_stack
from ultralytics import YOLO
import cv2
from ultralytics.utils.ops import xyxy2xywh,clip_boxes,xywh2xyxy
from ultralytics.utils.metrics import box_iou
import torch
import numpy  as np
import nibabel as nib
from tqdm import tqdm
import sys

anno_dirs="/mnt/hdd1/zhulu/blood_stage2/PNG-1.14"
swi="swi"
pha="phase"

gain,pad=1.02,10  #原医学图像图只有一个通道，是灰白图
BGR=True
reshape_size=16

fixed_width = 16  # 设置固定宽度
fixed_height = 16 # 设置固定高度

all_matches=[]

match_CMB=0
all_CMB=0
non_cmb=0
miss_CMB=0

write_stack_path="/mnt/hdd1/zhulu/blood_stage2/blood_anno/stackdata"
coco="swi/label/COCO"

for patient_name in tqdm(os.listdir(anno_dirs)): #得到每个病人全部的swi图像和label以及phase图像    
    second_flag=False
    
    patient_dir=os.path.join(anno_dirs,patient_name)
    swi_imgs,phase_imgs=get_patient_data(patient_dir)
    swi_imgs=sort_files(swi_imgs)
    phase_imgs=sort_files(phase_imgs)
    
    imageSwi_dir=os.path.join(patient_dir,swi)
    imagePhase_dir=os.path.join(patient_dir,pha)
    
    coco_json=os.path.join(patient_dir,coco,"annotations.json")
    with open(coco_json, 'r') as file:
        datas=json.load(file)
    
    annotations=datas["annotations"]
    
    #建立image_id和image_name的映射，便于annotation找到对应的图像
    img_infos=datas["images"]
    img_id2name={}
    for img_info in img_infos:
        img_id=img_info["id"]
        img_name=img_info["file_name"]
        img_id2name[img_id]=[img_name,0] #第二个位置为count表示同一张图像上已出现的框的个数
        
    
    for annotation in annotations:
        img_id=annotation["image_id"]
        image_name=img_id2name[img_id][0]
        layer_num=image_name.split("-")[-1].split(".")[0]
        index=int(layer_num)-1
        anno_box=annotation["bbox"]
        x1,y1,w,h=anno_box
        x2=x1+w
        y2=y1+h
        second_flag=False
        image_swis,image_phases=img_stack(imageSwi_dir,imagePhase_dir,swi_imgs,phase_imgs,index,second_flag)
        
        crop_swi = image_swis[int(y1) : int(y2), int(x1) : int(x2), :: (1 if BGR else -1)]
        crop_phase=image_phases[int(y1) : int(y2), int(x1) : int(x2), :: (1 if BGR else -1)]
        #没有数据归一化操作
        crop_Data=np.hstack((crop_swi,crop_phase))
        
        if annotation["category_id"]==3: #如果是人工标注，暂时不管
            continue
        
        if crop_Data.shape!=(16,32,16):  # 在yolo输出时已经确保框的大小为16，16
            print(f"error {patient_name}-{image_name}-{crop_Data.shape}") # 没有维度错误的样本
            sys.exit()
        
        if 0 not in crop_Data.shape: #创建nii文件
            scaling = np.eye(4)
            nii_img = nib.Nifti1Image(crop_Data, scaling)  #crop_data的数据值不会受到scaling的影响
            nii_img.header['descrip'] = f'Bounding boxes: {[x1,y1,x2,y2]}'  # 将坐标信息也存放到nii文件中

        if annotation["category_id"]==1: #出血点
            j=img_id2name[img_id][1]
            nii_file_name=f'/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add3/CMB/Patient-{patient_name}-{layer_num}-{j}.nii'
            nib.save(nii_img,nii_file_name)  
            img_id2name[img_id][1]+=1
        elif annotation["category_id"]==2: #非出血点，如果值等于3就是人工标注也是出血点
            j=img_id2name[img_id][1]
            nii_file_name=f'/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add3/Non_CMB/Patient-{patient_name}-{layer_num}-{j}.nii'
            nib.save(nii_img,nii_file_name)  
            img_id2name[img_id][1]+=1
        



