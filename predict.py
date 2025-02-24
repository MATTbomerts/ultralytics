from ultralytics.utils.ops import xyxy2xywh,clip_boxes,xywh2xyxy
from ultralytics import YOLO
import cv2
import torch

gain,pad,shape=1.02,10,(512,512,3)  #原医学图像图只有一个通道，是灰白图
BGR=True
fixed_width=16
fixed_height=16

image="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/Patient-0018847234/Series-Ax SWAN new/img-00005-00014.png"
model = YOLO("runs/detect/train32/weights/best.pt")  
image_swi=cv2.imread(image)

results = model.predict(image_swi,conf=0.01,iou=0.1,max_det=5,verbose=False) #第一阶段通过三通道进行预测，第二阶段转换为单通道灰度图预测
#根据层图像获取box信息
for i, result in enumerate(results): #单张图像处理的话，i始终是0，因为只有一张图像
    for j, box in enumerate(result.boxes): # j表示是第几个box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        b = xyxy2xywh(box.xyxy[0].view(-1, 4))  # boxes #将原来的xyxy格式转换为xywh格式
        b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad 将box的宽和高乘以gain再加上pad，中心点保持不变
        
        #获得目标的边界框
        xyxy = xywh2xyxy(b).long()   #结果是x表示列数，y表示行数
        
        center_x = (xyxy[0,0] + xyxy[0,2]) / 2  # (x1 + x2) / 2
        center_y = (xyxy[0,1] + xyxy[0,3]) / 2  # (y1 + y2) / 2
        
        # 计算新的边界框坐标，限制yolo输出框的大小
        new_x1 = int(center_x - fixed_width / 2)
        new_y1 = int(center_y - fixed_height / 2)
        new_x2 = int(center_x + fixed_width / 2)
        new_y2 = int(center_y + fixed_height / 2)

        # xyxy_fixed = torch.tensor([[new_x1, new_y1, new_x2, new_y2]])
        
        xyxy = clip_boxes(xyxy,shape) #保证扩充之后不会超出原图的边界
        
        crop_swi = image_swi[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]

        #蓝色，rectangle 参数坐标为point (列，行)
        cv2.rectangle(image_swi, (int(xyxy[0, 0]), int(xyxy[0, 1])), (int(xyxy[0, 2]), int(xyxy[0, 3])), color=(255,0,0), thickness=1)
        cv2.imwrite("temp/temp.png",image_swi)