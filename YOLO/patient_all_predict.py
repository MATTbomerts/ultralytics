from ultralytics import YOLO
import cv2
import numpy as np
import os
from ultralytics.utils.ops import xyxy2xywh,clip_boxes,xywh2xyxy
import re
import nibabel as nib
import numpy as np
from tqdm import tqdm

gain,pad,shape=1.02,10,(512,512,3)  #原医学图像图只有一个通道，是灰白图
BGR=True
reshape_size=16


patient_name="Patient-0002162909"
patient=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient_name}/Series-Ax SWAN new"
patient_imgs=[file for file in os.listdir(patient) if file.endswith(".png")]

label="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/Patient-0002162909/label"
label_files=[file for file in os.listdir(label) if file.endswith(".txt")]
model = YOLO("runs/detect/train20/weights/best.pt")  # 替换为实际的模型文件路径

for img in tqdm(patient_imgs):
    image_swi=cv2.imread(os.path.join(patient,img)) #读取单张图像，不是灰度图，512，512，3
    image_swi_draw=image_swi.copy()
    height,width=image_swi.shape[0],image_swi.shape[1]
    #一个results会预测出很多个box，但是都是对于这一层而言
    results = model.predict(image_swi,conf=0.01,iou=0.1,imgsz=512,max_det=5,verbose=False)  
    #绘制标注结果,找相同层面
    imglayer=img.split("-")[-1].split(".")[0]
    label_file=[file for file in label_files if imglayer in file]
    if label_file:
        with open(os.path.join(label,label_file[0]), 'r') as file:
            lines = file.readlines()
            for line in lines:
                #center_x表示列，center_y表示中心行
                id,center_x,center_y,width_norm,height_norm=line.strip().split()
                # center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), 20, 20
                center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), round(float(width_norm)*width), round(float(height_norm)*height)
                x1=center_x-w//2 #x表示的列，y表示的行，从txt中的中心转换为，现在h,w用的没错
                y1=center_y-h//2
                x2=center_x+w//2
                y2=center_y+h//2
                cv2.rectangle(image_swi_draw, (x1, y1), (x2, y2), color=(255,0,0), thickness=1) # 蓝色
                
        
    #绘制预测结果框
    for i, result in enumerate(results): #单张图像处理的话，i始终是0，因为只有一张图像
        for j, box in enumerate(result.boxes): # j表示是第几个box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            b = xyxy2xywh(box.xyxy[0].view(-1, 4))  # boxes #将原来的xyxy格式转换为xywh格式
            b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad 将box的宽和高乘以gain再加上pad，中心点保持不变
            #获得目标的边界框
            xyxy = xywh2xyxy(b).long() 
            xyxy = clip_boxes(xyxy,shape) #保证扩充之后不会超出原图的边界
            cv2.rectangle(image_swi_draw, (int(xyxy[0,0]), int(xyxy[0,1])), (int(xyxy[0,2]), int(xyxy[0,3])), (0, 0, 255), 1)

    write_path=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/predict_Result/{patient_name}"
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    img_Res_path=os.path.join(write_path,img)
    cv2.imwrite(img_Res_path, image_swi_draw)
    
            