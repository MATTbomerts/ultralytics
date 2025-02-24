from ultralytics import YOLO
import cv2
import numpy as np
import os
from ultralytics.utils.ops import xyxy2xywh,clip_boxes,xywh2xyxy
import re
import nibabel as nib
from secondStage.T3D_cnn import NiiDataset2,val_model,CNN3D
from torch.utils.data import DataLoader, Dataset
import torch
from ultralytics.utils.metrics import box_iou
import torchvision.transforms as transforms

second_devices=["Patient-0000406857","Patient-0000589446",'Patient-0000745885','Patient-0003537968'
                ,'Patient-0010134951','Patient-0018227242','Patient-0018463963']


fixed_width = 16  # 设置固定宽度
fixed_height = 16 # 设置固定高度

def sort_files(file_list):
    # 正则表达式提取文件名中的数字部分
    def extract_number(filename):
        match = re.search(r"-(\d+)\.png", filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # 如果没有匹配到数字，返回一个很大的数字以便排序到最后

    # 按照提取的数字部分排序文件名
    sorted_files = sorted(file_list, key=extract_number)
    return sorted_files

# 加载YOLOv8模型
model = YOLO("runs/detect/train34/weights/best.pt")  
output_dir = "/mnt/hdd1/zhulu/hospital/cropped_imgs"
# os.makedirs(output_dir, exist_ok=True)

 #从第一阶段验证集中读取不同病人的数据，再跑到对应病人的原始图像数据文件中获取3d数据
 #为了加快数据的处理，先把验证集中的数据按照病人划分到一起
 #但是第一阶段测试集已经变了，加入了第二阶段的病人了，因此要排除一下
input_img_dir="/mnt/hdd1/zhulu/hospital/images/val"   
input_label_dir="/mnt/hdd1/zhulu/hospital/labels/val"

input_imgs=os.listdir(input_img_dir)

#region
#dict中每一个key对应一个病人，其值表示病人微出血点所在的切面  old方法
#键：Patient-0000284759；值：[Patient-0000284759_img-00005-00067.png  ...]
# patient_dict=dict()
# for img in input_imgs:
#     patient_num=img.split("_")[0]
#     if patient_num not in patient_dict:
#         patient_dict[patient_num]=[]
#         patient_dict[patient_num].append(img)
#     else:
#         patient_dict[patient_num].append(img)
#endregion

""" 所有层面图像 """
#先测一个病人,以病人为单位预测
patient_dict=dict()

#region
# patient_img_path="/mnt/hdd1/zhulu/hospital/Patient-0000808485/HD-BET/SWI"  #这是第一阶段的数据
# patient_imgs=os.listdir(patient_img_path)
# patient_imgs=sort_files(patient_imgs) #根据图像的切片号进行排序，后面才能直接根据index进行上下层读取
# patient_dict["Patient-0000808485"]=patient_imgs  #切面没有进行排序

# patient_img_path="/mnt/hdd1/zhulu/blood_stage2/blood_anno2/PNG/0001232716/swi"
# patient_imgs=[img for img in os.listdir(patient_img_path) if img.endswith(".png")]
# patient_imgs=sort_files(patient_imgs) #根据图像的切片号进行排序，后面才能直接根据index进行上下层读取
# patient_dict["0001232716"]=patient_imgs  #切面没有进行排序
#endregion


val_patients=["Patient-0018699418","Patient-0018847234","Patient-0019014525","Patient-0019639742"]

patient_dirs="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
for patient_dir in os.listdir(patient_dirs):
    if patient_dir not in val_patients:
        continue
    patient_img_path=os.path.join(patient_dirs,patient_dir,"Series-Ax SWAN new")
    patient_imgs=[img for img in os.listdir(patient_img_path) if img.endswith(".png")]
    patient_imgs=sort_files(patient_imgs) #根据图像的切片号进行排序，后面才能直接根据index进行上下层读取
    patient_dict[patient_dir]=patient_imgs  #切面没有进行排序


# 模型加载
model_3D=CNN3D(2).cuda()

gain,pad,shape=1.02,10,(512,512,3)  #原医学图像图只有一个通道，是灰白图
BGR=True
reshape_size=16


def patient_build_nii(patient_dict,model):
    all_data=[]
    count=0  #表示一共有多少层图像
    truth_count=0  #表示有多少框命中，会多于count，因为一个层面可能有多个出血点
    ground_truth_count=0
    false_count=0
    for patient,imgs in patient_dict.items():
        # 第一阶段的训练数据
        # /mnt/hdd1/zhulu/hospital/Patient-0000808485/HD-BET/SWI
        imageSwi_dir=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/Series-Ax SWAN new"
        # imageSwi_dir = f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/Series-Ax SWAN new" 
        # /mnt/hdd1/zhulu/hospital/Patient-0000808485/HD-BET/PHASE
        imagePhase_dir=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/Series-FILT_PHA_ Ax SWAN new"
        # imagePhase_dir = f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/Series-FILT_PHA_ Ax SWAN new"
        label_dir=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/label"  
        
        #知道当前病人出血点所在的层面，直接+或-就可以了，不需要排序吧 
        #确保数据完整性无误
        if not os.path.exists(imageSwi_dir) or not os.path.exists(imagePhase_dir): # 因为使用了第二阶段的数据，但是第一阶段的原始数据目录下没有第二阶段文件，所以排除一下
            continue
        
        Swi_imgs_len=len(os.listdir(imageSwi_dir))
        Phase_imgs_len=len(os.listdir(imagePhase_dir))
        if Phase_imgs_len==0:
            print("empty: "+ imagePhase_dir)
            continue
            # return
        if (Swi_imgs_len-1)!=Phase_imgs_len: # 在新的数据中两个文件夹下不会有其他的内容,导致数量不一致
            print("error: "+ imageSwi_dir)
            continue
            # return

        depth=16
        # img_prefix="-".join(imgs[0].split("-")[:2])
        #针对于每个层面都要进行出血点的检测和3D nii文件的构造
        for index,img in enumerate(imgs):  #此处是一个病人存在标注的所有层面，此时每一张图像都要进行预测
            count+=1
            ground_truth_boxes=[]
            label_path=img.split(".")[0]+".txt"
            label_file=os.path.join(label_dir,label_path)
            image_swi=cv2.imread(os.path.join(imageSwi_dir,img))
            image_swi_draw=image_swi.copy()  #进行图像绘制，在image_swi_draw上的操作不会影响image_Swi
            height,width=image_swi.shape[0],image_swi.shape[1]  #按照矩阵来读尺寸（行，列）
            if height!=512 or width!=512:
                print("error: ",img,image_swi.shape)
                
            
            if not os.path.exists(label_file): #如果对应的层面没有txt标注文件，那么就是没有出血点
                 ground_truth_boxes=[] #如果没有出血点，那么这层图像预测的结果就是全错的，应该标记为0(Non-CMB)
            
            else: #真实预测时,并没有ground truth txt文件,因此全部第一阶段的预测人工标签处都是0
                with open(label_file, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        #center_x表示列，center_y表示中心行
                        id,center_x,center_y,width_norm,height_norm=line.strip().split()  #这里拿到的是压缩比，归一化的结果
                        
                        # width=512  #第一阶段的图像都是512*512的规模，应该直接用原本读取的图像的尺寸来计算
                        # height=512
                        #貌似下面计算错误？？ 
                        #从txt中拿到的一行数据，x_center(与width相关)，y_center(与height相关)，w(与width相关)，h(与height相关)
                        # w和h都是20的话，下面的去中心化操作就没有问题
                        center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), 20, 20
                        # center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), round(float(width_norm)*width), round(float(height_norm)*height)
                        x1=center_x-w//2 #x表示的列，y表示的行，从txt中的中心转换为，现在h,w用的没错
                        y1=center_y-h//2
                        x2=center_x+w//2
                        y2=center_y+h//2
                        if id==str(0):
                            ground_truth_boxes.append([x1,y1,x2,y2])  # box的顺序应该是什么？
                            cv2.rectangle(image_swi_draw, (x1, y1), (x2, y2), color=(0,0,255), thickness=1)
            
            #这里的ground truth 文件中读出的记录全是出血点
            ground_truth_count+=len(ground_truth_boxes)  
            
            results = model.predict(image_swi,conf=0.01,iou=0.1,max_det=5,verbose=False) #第一阶段通过三通道进行预测，第二阶段转换为单通道灰度图预测
            image_swi = cv2.cvtColor(image_swi, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
            
            layer_num=img.split("-")[-1].split(".")[0]
            phase_sequence=os.listdir(imagePhase_dir)[0].split("-")[1] #得到相位图的序列号
            
            image_swis=[]
            
            image_phases=[]
            
            if len(results[0].boxes)==0:  #如果当前层没有预测出box，则直接跳过后面的处理
                continue
            #如果该层图前面还有8层（包括当前层），后面还有8层，那么就可以进行处理，
            if index-7>=0 and index+8<=len(imgs)-1:#index就是该图像在所有图像中的按层数排序的位置
                for i in range(depth):  #验证没有问题
                    Simage_depth_path=imgs[index-7+i]  
                    image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                    image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                    image_swis.append(image_Sdepth_data)
                    
                    deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                    img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                    image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                    if patient in second_devices:  #进行不同设备的统一操作
                        image_Pdepth_data=255-image_Pdepth_data
                    image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                    image_phases.append(image_Pdepth_data)
                    
            else:
                if index-7<0: #当前层以及前面层不足8，后面层多取
                    for i in range(depth): 
                        Simage_depth_path=imgs[i]  # 直接从前向后遍历16层
                        image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                        image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                        image_swis.append(image_Sdepth_data)
                        
                        #取出所有层数，去找对应的相位图
                        deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                        img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                        image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                        
                        if patient in second_devices: #将第一层相位图取反，接下来拼接层也进行取反
                            image_Pdepth_data=255-image_Pdepth_data
                        
                        image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                        image_phases.append(image_Pdepth_data)
            
                elif index+8>len(imgs)-1: #后面层不足8，前面层多取，直接从后向前定位16层
                    start=len(imgs)-1-15
                    for i in range(start,len(imgs)): # 表示从start到len(imgs_path)-1
                        Simage_depth_path=imgs[i]  # 直接从前向后遍历16层
                        image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                        image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                        image_swis.append(image_Sdepth_data)
                        
                        deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                        img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                        image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                        
                        if patient in second_devices: #将第一层相位图取反，接下来拼接层也进行取反
                            image_Pdepth_data=255-image_Pdepth_data
                        
                        image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                        image_phases.append(image_Pdepth_data)
            
            image_swi=np.stack(image_swis,axis=2)  #现在得到的是完整的图像，没有经过裁剪   
            image_phase=np.stack(image_phases,axis=2)  #现在得到的是完整的图像，没有经过裁剪  
            if (image_swi.shape[2]!=16):
                print("error depth :",img,imageSwi_dir,image_swi.shape)
            if (image_phase.shape!=image_swi.shape):
                print("error shape:",img,imagePhase_dir,image_phase.shape)
            
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

                    xyxy_fixed = torch.tensor([[new_x1, new_y1, new_x2, new_y2]])
                    
                    xyxy = clip_boxes(xyxy_fixed,shape) #保证扩充之后不会超出原图的边界
                    
                    crop_swi = image_swi[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
                    crop_phase=image_phase[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
                    #蓝色，rectangle 参数坐标为point (列，行)
                    cv2.rectangle(image_swi_draw, (int(xyxy[0, 0]), int(xyxy[0, 1])), (int(xyxy[0, 2]), int(xyxy[0, 3])), color=(255,0,0), thickness=1)
                    #没有数据归一化操作
                    crop_Data=np.hstack((crop_swi,crop_phase))  #看一下是否是在第二个维度上进行的拼接

                    """ 这里是进行crop_yolo之后为预测框打标签 出血点为真 """
                    #ground_truth_boxes（列行列行），xyxy(列行列行)一致，ground_truth_boxes只有出血点
                    if len(ground_truth_boxes)==0:  #在真实预测时都是走的这一步
                        false_count+=1
                        file_name=f"/mnt/hdd1/zhulu/hospital/second_stage/val_all/Non-CMB/{patient}-{layer_num}-{j}"
                        all_data.append([crop_Data,0,file_name,xyxy])
                    else:
                        iou=box_iou(torch.tensor(ground_truth_boxes,device=xyxy.device),xyxy) #ground_truth_boxes是xyxy格式
                        matches = (iou>0.2).any()
                        
                        if (matches):
                            # 在ground_truth_boxes中找到匹配的预测结果之后，就将该真实值删除
                            # 因此最终的truth_count就是预测的匹配了多少的真实框
                            truth_count+=1
                            #j表示yolo预测出来的第几个框（但也不清楚是哪一个框）
                            file_name=f"/mnt/hdd1/zhulu/hospital/second_stage/val_all/CMB/{patient}-{layer_num}-{j}"
                            all_data.append([crop_Data,1,file_name,xyxy])
                            # nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/val_all/CMB/{patient}-{layer_num}-{j}.nii')
                        else:  #非出血点
                            false_count+=1
                            file_name=f"/mnt/hdd1/zhulu/hospital/second_stage/val_all/Non-CMB/{patient}-{layer_num}-{j}"
                            all_data.append([crop_Data,0,file_name,xyxy])
            cv2.imwrite("temp/yolo_crop.png", image_swi_draw)
                        # nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/val_all/Non-CMB/{patient}-{layer_num}-{j}.nii')

    transform = transforms.Compose([
    transforms.ToTensor(),  #操作会进行数据归一化
    transforms.Lambda(lambda x: x.permute(1, 2, 0))  #但是好像结合3D-CNN这个维度不需要转换
    ])

    val_dataset=NiiDataset2(all_data,transform) #加入归一化
    valid_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    model_3D.load_state_dict(torch.load(f"parameters/swi/CMB_3DCNN_norm_lr0001_91.pth"))
    # model_3D.load_state_dict(torch.load(f"parameters/{resolution}/CMB_3DCNN_new4.pth"))
    val_model(model_3D,valid_loader)
    
    print("truth_count:",truth_count)  #表示yolo预测出来的框中有多少是真的
    print("ground_truth:",ground_truth_count)
    print("false_count:",false_count) #表示yolo预测出来的框中有多少是假的


patient_build_nii(patient_dict,model)


#在实际预测过程中,不关心真实标签,随便设置,只关心预测结果