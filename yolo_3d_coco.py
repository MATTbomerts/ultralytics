import os
from my_tool.utils import get_patient_data,sort_files
from ultralytics import YOLO
import cv2
from ultralytics.utils.ops import xyxy2xywh,clip_boxes,xywh2xyxy
from ultralytics.utils.metrics import box_iou
import torch
import numpy  as np
from tqdm import tqdm
import json

def box_to_segmentation(box):
    """
    Convert a bounding box to segmentation.
    
    Args:
        box (list or tuple): [x, y, width, height]
    
    Returns:
        list: Segmentation as a list of points [x1, y1, x2, y2, ...].
    """
    x, y, width, height = box
    segmentation = [
        x, y,                # Top-left
        x + width, y,        # Top-right
        x + width, y + height,  # Bottom-right
        x, y + height        # Bottom-left
    ]
    return segmentation

patient_dirs="/mnt/hdd1/zhulu/blood_stage2/blood_anno4/PNG"
patients=[patient for patient in os.listdir(patient_dirs)  if patient[0].isdigit()]

swi="swi"
pha="phase"


gain,pad,shape=1.02,10,(512,512,3)  #原医学图像图只有一个通道，是灰白图
BGR=True
reshape_size=16

fixed_width = 16  # 设置固定宽度
fixed_height = 16 # 设置固定高度

# model = YOLO("runs/detect/train34/weights/best.pt")  # 替换为实际的模型文件路径
model = YOLO("runs/detect/train34/weights/best.pt")  # 替换为实际的模型文件路径
all_matches=[]

match_CMB=0
all_CMB=0
non_cmb=0
miss_CMB=0

write_stack_path="/mnt/hdd1/zhulu/blood_stage2/blood_anno/stackdata"

for patient_name in tqdm(patients): #得到每个病人全部的swi图像和label以及phase图像
    
    coco_dict={
        "images":[],
        "annotations":[],
        "categories": [
            {
                "id": 1, # 1表示出血点，2表示非出血点
                "name": "",
                "color": [  #橙色
                    244,
                    108,
                    59
                ],
                "supercategory": ""
            },
            {
                "id": 2,
                "name": "",
                "color": [  #黄色
                    255,
                    255,
                    0
                ],
                "supercategory": ""
            },
            {
                "id": 3,
                "name": "",
                "color": [  #green
                    0,
                    255,
                    0
                ],
                "supercategory": ""
            }
        ],
    }
    
    out_put_path=os.path.join(patient_dirs,patient_name,"swi","label","COCO")
    if os.path.exists(out_put_path)==False:
        os.makedirs(out_put_path)
    
    patient_dir=os.path.join(patient_dirs,patient_name)
    
    if os.path.exists(os.path.join("/mnt/hdd1/zhulu/blood_stage2/DATA3/PNG",patient_name,"label"))==False:
        print("没有label: ",patient_name)
        continue
    
    swi_imgs,_,phase_imgs=get_patient_data(patient_dir)
    swi_imgs=sort_files(swi_imgs)
    phase_imgs=sort_files(phase_imgs)
    
    imageSwi_dir=os.path.join(patient_dir,swi)
    
    for index,swi_img_file in enumerate(swi_imgs): #index和图像的层数存在1的差别
        
        img_flag=0 #表示在coco标注中是否已经有这个图像了
        layer_num=swi_img_file.split("-")[-1].split(".")[0]
        ground_truth_boxes=[]
        swi_img=cv2.imread(os.path.join(imageSwi_dir,swi_img_file))
        
        image_swi_draw=swi_img.copy()
        deep_layer_num=swi_img_file.split("-")[-1].split(".")[0]
        

        results = model.predict(swi_img,conf=0.001,iou=0.1,imgsz=512,max_det=5,verbose=False)
        label_name=swi_img_file.split(".")[0]+".txt"
        label_file=os.path.join(patient_dir.replace("blood_anno4","DATA3"),"label",label_name)
        height,width=swi_img.shape[0],swi_img.shape[1]
        if os.path.exists(label_file): #如果当前图层存在标注，读出标注框
            with open(label_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    #center_x表示列，center_y表示中心行
                    id,center_x,center_y,width_norm,height_norm=line.strip().split()  #这里拿到的是压缩比，归一化的结果
                    center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), 20, 20
                    # center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), round(float(width_norm)*width), round(float(height_norm)*height)
                    x1=center_x-w//2 #x表示的列，y表示的行，从txt中的中心转换为，现在h,w用的没错
                    y1=center_y-h//2
                    x2=center_x+w//2
                    y2=center_y+h//2
                    if id=="0": #出血点
                        ground_truth_boxes.append([x1,y1,x2,y2])  # box的顺序应该是什么？
                        cv2.rectangle(image_swi_draw, (x1, y1), (x2, y2), color=(0,0,255), thickness=1) #蓝色
        ground_truth_boxes_copy=ground_truth_boxes.copy()
        if not results[0].boxes:  #如果当前层没有预测出box，则直接跳过，下面都是有预测
            continue
        all_CMB+=len(ground_truth_boxes)
        
        image_id=len(coco_dict["images"])+1 # 在预测点外面定义，保证同一张图像的多个预测点，不会改变image_id
        
        # 每一个点都会有标注
        for i, result in enumerate(results): #单张图像处理的话，i始终是0，因为只有一张图像
            for j, box in enumerate(result.boxes): # j表示是第几个box
                # x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                b = xyxy2xywh(box.xyxy[0].view(-1, 4))  # boxes #将原来的xyxy格式转换为xywh格式
                b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad 将box的宽和高乘以gain再加上pad，中心点保持不变
                #获得目标的边界框
                xyxy = xywh2xyxy(b).long()    #左上角和右下角
                # 计算原始框的中心点  固定输出框的大小
                center_x = (xyxy[0,0] + xyxy[0,2]) / 2  # (x1 + x2) / 2
                center_y = (xyxy[0,1] + xyxy[0,3]) / 2  # (y1 + y2) / 2
                
                # 计算新的边界框坐标
                new_x1 = int(center_x - fixed_width / 2)
                new_y1 = int(center_y - fixed_height / 2)
                new_x2 = int(center_x + fixed_width / 2)
                new_y2 = int(center_y + fixed_height / 2)

                # 更新坐标
                xyxy_fixed = torch.tensor([[new_x1, new_y1, new_x2, new_y2]])
                
                xyxy = clip_boxes(xyxy_fixed,shape) #保证扩充之后不会超出原图的边界
                cv2.rectangle(image_swi_draw, (int(xyxy[0,0]), int(xyxy[0,1])), (int(xyxy[0,2]), int(xyxy[0,3])), (0, 0, 255), 1)
                
                #将左上角和右下角坐标转换为左上角和宽高
                xywh=[xyxy[0,0],xyxy[0,1],xyxy[0,2]-xyxy[0,0],xyxy[0,3]-xyxy[0,1]]
                xywh = [int(t.item()) for t in xywh]
              
                if len(ground_truth_boxes)!=0: #如果标注中没有出血点，那么预测出来的结果都是非出血点
                    
                    iou=box_iou(torch.tensor(ground_truth_boxes,device=xyxy.device),xyxy) #ground_truth_boxes是xyxy格式
                    iou=iou.squeeze()
                    matches = (iou>0.2).any()
                    
                    if (matches): #如果与真实标签匹配上，说明当前的预测正确就是出血点
                        category=1 #出血点标签
                        max_iou_value, max_iou_index = torch.max(iou, dim=0)  # 最后是选择最大的iou标注框，0.2只是一个下限
                        result_index = max_iou_index.item()
                        ground_truth_boxes = [ground_truth_boxes[i] for i in range(len(ground_truth_boxes)) if i != result_index]
                        
                    else:  #非出血点
                        category=2 #非出血点标签
                else: #如果没有人工标注，那么预测出来的都是非出血点
                    category=2
                
                segmentation_pcoints = box_to_segmentation(xywh)
                
                # 添加图像信息，如果当前图像有预测
                if img_flag==0:
                    coco_dict["images"].append({
                        "id":image_id,
                        "file_name": swi_img_file,  # 替换为你的图片文件名
                        "width": width,
                        "height": height
                    })
                    img_flag=1
                    
                # 先有标注再有点标注
                coco_dict["annotations"].append({
                    "id": len(coco_dict["annotations"])+1, #annotation_id从1开始
                    "image_id": image_id,  #image_id从1开始
                    "category_id": category, #1表示出血点，0表示非出血点
                    "bbox": xywh,  # 转换后的 COCO 格式 bbox
                    "segmentation":[segmentation_pcoints],  #转换后的segmentation
                    "area": xywh[2] * xywh[3],  # 计算面积
                    })
        cv2.imwrite("temp/2.png", image_swi_draw)
        
        if len(ground_truth_boxes_copy)!=0:  #真实的人工标注
            for box in ground_truth_boxes_copy:
                xywh=[box[0],box[1],box[2]-box[0],box[3]-box[1]]
                xywh = [int(t) for t in xywh]
                segmentation_pcoints = box_to_segmentation(xywh)
                # 先有标注再有点标注
                coco_dict["annotations"].append({
                    "id": len(coco_dict["annotations"])+1, #annotation_id从1开始
                    "image_id": image_id,  #image_id从1开始
                    "category_id": 3, #1表示出血点，0表示非出血点
                    "bbox": xywh,  # 转换后的 COCO 格式 bbox
                    "segmentation":[segmentation_pcoints],  #转换后的segmentation
                    "area": xywh[2] * xywh[3],  # 计算面积
                    })
                
                    
    # 保存 COCO 数据
    if len(coco_dict["images"])>0:
        with open(os.path.join(out_put_path,"annotations.json"), 'w') as f:
            json.dump(coco_dict, f, indent=4)
            
                   
                            

