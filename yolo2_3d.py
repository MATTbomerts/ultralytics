import os
from my_tool.utils import get_patient_data,sort_files,img_stack
from ultralytics import YOLO
import cv2
from ultralytics.utils.ops import xyxy2xywh,clip_boxes,xywh2xyxy
from ultralytics.utils.metrics import box_iou
import torch
import numpy  as np
import nibabel as nib
from tqdm import tqdm


resolution="high"

patient_dirs="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
swi="Series-Ax SWAN new"
pha="Series-FILT_PHA_ Ax SWAN new"
train_patient_dir="/mnt/hdd1/zhulu/hospital/images/high_resolution/train"
#第一阶段数据集下一共有哪些病人
train_patient_imgs=[img.split("_")[0] for img in os.listdir(train_patient_dir)] 
train_patient=set(train_patient_imgs)

second_devices=["Patient-0000406857","Patient-0000589446",'Patient-0000745885','Patient-0003537968'
                ,'Patient-0010134951','Patient-0018227242','Patient-0018463963']


gain,pad,shape=1.02,10,(512,512,3)  #原医学图像图只有一个通道，是灰白图
BGR=True
reshape_size=16

fixed_width = 16  # 设置固定宽度
fixed_height = 16 # 设置固定高度

model = YOLO("runs/detect/train32/weights/best.pt")  # 替换为实际的模型文件路径
all_matches=[]

match_CMB=0
all_CMB=0
non_cmb=0
miss_CMB=0

write_stack_path="/mnt/hdd1/zhulu/blood_stage2/blood_anno/stackdata"

for patient_name in tqdm(train_patient): #得到每个病人全部的swi图像和label以及phase图像
    if os.path.exists(os.path.join("/mnt/hdd1/zhulu/anno_box/train",patient_name))==False:
        os.makedirs(os.path.join("/mnt/hdd1/zhulu/anno_box/train",patient_name))
        
    second_flag=False
    if patient_name in second_devices:
        second_flag=True
    patient_dir=os.path.join(patient_dirs,patient_name)
    swi_imgs,labels,phase_imgs=get_patient_data(patient_dir)
    swi_imgs=sort_files(swi_imgs)
    phase_imgs=sort_files(phase_imgs)
    
    imageSwi_dir=os.path.join(patient_dir,swi)
    imagePhase_dir=os.path.join(patient_dir,pha)
    for index,swi_img_file in enumerate(swi_imgs): #index和图像的层数存在1的差别
        layer_num=swi_img_file.split("-")[-1].split(".")[0]
        ground_truth_boxes=[]
        swi_img=cv2.imread(os.path.join(imageSwi_dir,swi_img_file))
        
        image_swi_draw=swi_img.copy()
        deep_layer_num=swi_img_file.split("-")[-1].split(".")[0]
        
        phase_sequence=os.listdir(imagePhase_dir)[0].split("-")[1]
        img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
        image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
        
        results = model.predict(swi_img,conf=0.01,iou=0.1,imgsz=512,max_det=5,verbose=False)
        label_name=swi_img_file.split(".")[0]+".txt"
        label_file=os.path.join(patient_dir,"label",label_name)
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
        
        if not results[0].boxes:  #如果当前层没有预测出box，则直接跳过
            continue
        all_CMB+=len(ground_truth_boxes)
        
        image_swis,image_phases=img_stack(imageSwi_dir,imagePhase_dir,swi_imgs,phase_imgs,index,second_flag)
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
                
                # cv2.rectangle(image_swi_draw, (int(xyxy[0,0]), int(xyxy[0,1])), (int(xyxy[0,2]), int(xyxy[0,3])), (0, 0, 255), 1)
                # cv2.rectangle(image_Pdepth_data, (int(xyxy[0,0]), int(xyxy[0,1])), (int(xyxy[0,2]), int(xyxy[0,3])), (0, 0, 255), 1)
                
                crop_swi = image_swis[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
                crop_phase=image_phases[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
                #没有数据归一化操作
                crop_Data=np.hstack((crop_swi,crop_phase))
                if crop_Data.shape!=(16,32,16):  # 在yolo输出时已经确保框的大小为16，16
                    print("error") # 没有维度错误的样本
                
                if 0 not in crop_Data.shape: #创建nii文件
                    scaling = np.eye(4)
                    nii_img = nib.Nifti1Image(crop_Data, scaling)  #crop_data的数据值不会受到scaling的影响
                    nii_img.header['descrip'] = f'Bounding boxes: {xyxy.squeeze().tolist()}'  # 将坐标信息也存放到nii文件中
                
                #ground_truth_boxes（列行列行），xyxy(列行列行)一致，ground_truth_boxes只有出血点
                if len(ground_truth_boxes)==0: #如果标注中没有出血点，那么预测出来的结果都是非出血点
                    #将其保存到Non-CMB文件夹下
                    jk=1
                    non_cmb+=1  #/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all/Non-CMB
                    cv2.rectangle(image_swi_draw, (int(xyxy[0, 0]), int(xyxy[0, 1])), (int(xyxy[0, 2]), int(xyxy[0, 3])), color=(0,255,0), thickness=1) #绿色
                        # cv2.imwrite("2.png", image_swi_draw)
                    cv2.rectangle(image_Pdepth_data, (int(xyxy[0,0]), int(xyxy[0,1])), (int(xyxy[0,2]), int(xyxy[0,3])), (0,255,0), 1)
                    # nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/{resolution}/all/Non_CMB/{patient_name}-{layer_num}-{j}.nii')    
                else:
                    #计算每个预测框与所有标注框的iou值,box_iou的返回值维度为(N,M)，只要M
                    iou=box_iou(torch.tensor(ground_truth_boxes,device=xyxy.device),xyxy) #ground_truth_boxes是xyxy格式
                    iou=iou.squeeze()
                    matches = (iou>0.2).any()
                    # iou_values = iou[iou > 0.2] 
                    # cv2.rectangle(image_swi_draw, (int(xyxy[0, 0]), int(xyxy[0, 1])), (int(xyxy[0, 2]), int(xyxy[0, 3])), color=(255,0,0), thickness=1)
                    # cv2.imwrite(write_path, image_swi_draw)
                    if (matches): #如果与真实标签匹配上，说明当前的预测正确就是出血点
                        jk=1
                        match_CMB+=1
                        max_iou_value, max_iou_index = torch.max(iou, dim=0)  # 最后是选择最大的iou标注框，0.2只是一个下限
                        result_index = max_iou_index.item()
                        # remove_indices = (iou > 0.2).nonzero(as_tuple=True)[0] #这个返回的是所有的，并不只是一个值
                        ground_truth_boxes = [ground_truth_boxes[i] for i in range(len(ground_truth_boxes)) if i != result_index]
                        cv2.rectangle(image_swi_draw, (int(xyxy[0, 0]), int(xyxy[0, 1])), (int(xyxy[0, 2]), int(xyxy[0, 3])), color=(255,0,0), thickness=1) # 红色
                        cv2.rectangle(image_Pdepth_data, (int(xyxy[0,0]), int(xyxy[0,1])), (int(xyxy[0,2]), int(xyxy[0,3])), (255,0,0), 1)
                        
                        # all_matches.append(iou_values)
                        # nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/{resolution}/all/CMB/{patient_name}-{layer_num}-{j}.nii')    
                    else:  #非出血点
                        jk=1
                        non_cmb+=1
                        cv2.rectangle(image_swi_draw, (int(xyxy[0, 0]), int(xyxy[0, 1])), (int(xyxy[0, 2]), int(xyxy[0, 3])), color=(0,255,0), thickness=1) #绿色
                        # cv2.imwrite("2.png", image_swi_draw)
                        cv2.rectangle(image_Pdepth_data, (int(xyxy[0,0]), int(xyxy[0,1])), (int(xyxy[0,2]), int(xyxy[0,3])), (0,255,0), 1)
                        # nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/{resolution}/all//Non_CMB/{patient_name}-{layer_num}-{j}.nii')    
        if len(results[0].boxes)>0:
            stack_data=np.concatenate((image_swi_draw, image_Pdepth_data), axis=1)
            image_name=f"{patient_name}-{swi_img_file}"
            write_path=os.path.join(write_stack_path,image_name)
            cv2.imwrite(write_path, stack_data)
            # cv2.imwrite("1.png", stack_data)
                   
        if len(ground_truth_boxes)>0: #如果当前标注文本的标注框还剩，说明没有匹配完
            miss_CMB+=len(ground_truth_boxes)
        if miss_CMB+match_CMB!=all_CMB:  # 对这样图的所有预测框都判断完后
            print("error")
print(match_CMB/all_CMB)                              

lk=1+1



