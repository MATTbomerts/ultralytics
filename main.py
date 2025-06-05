
import cv2
import numpy as np
import os
import re
import nibabel as nib
from secondStage.T3D_cnn import NiiDataset2 as swi_NiiDataset2,predict_model,ComplexCNN3D as swi_CNN3D
from secondStage.T3D_cnn_phase import PredNiiDataset as phase_NiiDataset,ComplexCNN3D as phase_CNN3D,val_model as phase_val_model
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import  SimpleITK as sitk 
import argparse
import subprocess
import imageio
from pathlib import Path
from continue_code import process_dir


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="3D CMB Detection")
    parser.add_argument('--dicom_dir', type=str, required=True, help="Path to the dicom directory")
    parser.add_argument('--nii_dir', type=str, required=True, help="Path to save nii directory")
    parser.add_argument('--png_dir', type=str, required=True, help="Path to save png directory")
    
    return parser.parse_args()

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


#加载dicom文件
def load_dicom(dcm_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return image

#dicom之后的标准化
def normalize(original_image):
    new_spacing = [1.0, 1.0, 1.0] 
    original_size = original_image.GetSize()
    original_spacing = original_image.GetSpacing()
    
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    # resampler.SetSize(new_size)
    resampler.SetSize(original_size)
    resampler.SetOutputDirection(original_image.GetDirection())
    resampler.SetOutputOrigin(original_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(original_image.GetPixelIDValue())

    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_image = resampler.Execute(original_image)
    return resampled_image


def nii2img(file_path,output_dir):
    # file_path = "/mnt/hdd1/zhulu/blood_stage2/blood_anno/all_bet/0017691773.nii.gz"
    # output_dir = '/mnt/hdd1/zhulu/hospital/0017691773FILT/HD-BET/test'
    img = nib.load(file_path)
    img_fdata = img.get_fdata()
    # print("方向代码:", nib.aff2axcodes(img.affine))  # 应输出 ('L', 'P', 'S')
    
    
    # 获取图像的维度
    (x, y, z) = img.shape

    # 计算全局最小值和最大值
    global_min = np.min(img_fdata)
    global_max = np.max(img_fdata)

   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个切片，并保存为 PNG
    for i in range(z):
        # 选择切片并旋转
        silce = img_fdata[:, :, i]
        # cv2.imwrite("tmp.png", silce)  # 保存第一张切片作为示例
        
        silce = np.rot90(silce, k=-1) # 逆时针旋转 90 度
        # cv2.imwrite("tmp1.png", silce)  # 保存第一张切片作为示例
        
        # 全局归一化切片数据并转换为 uint8
        if global_max != global_min:
            slice_data = ((silce - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        else:
            slice_data = np.zeros_like(silce, dtype=np.uint8)  # 处理 max == min 的情况

        # 水平翻转
        slice_data = np.fliplr(slice_data)
        # cv2.imwrite("tmp2.png", slice_data)  # 保存第一张切片作为示例

        # 保存当前切片为 PNG 图片，统一设置swi图像的序列号为00005
        output_file = os.path.join(output_dir, 'img-00005-{0:05d}.png'.format(i + 1)) # 从编号1开始保存
        imageio.imwrite(output_file, slice_data)


def phase_mask_extract(mask_img_path,phase_img_path,output_dir):
    # 加载 phase 图像和 mask 图像  mask_img
    # mask_img = nib.load('/mnt/hdd1/zhulu/blood_stage2/blood_anno/all_bet/0019014525_mask.nii.gz')
    mask_img = nib.load(mask_img_path)
    # phase_img = nib.load('/mnt/hdd1/zhulu/blood_stage2/blood_anno/all_niigz/0019014525FILT.nii.gz')
    phase_img = nib.load(phase_img_path)
    # print("方向代码:", nib.aff2axcodes(phase_img.affine))  # 应输出 ('L', 'P', 'S')

    
    # # 获取图像数据
    phase_data = phase_img.get_fdata()
    mask_data = mask_img.get_fdata()
    
    # 应用mask
    masked_phase_data = phase_data * mask_data
    # 创建一个新的 Nifti1Image 对象
    # masked_phase_img = nib.Nifti1Image(masked_phase_data, phase_img.affine, phase_img.header)
    #原本的这个nii文件中仿射变换是有值的，但是在保存png图像时是不需要这个信息的
    # print(phase_img.affine)
    
    global_min = np.min(masked_phase_data)
    global_max = np.max(masked_phase_data)
    (x, y, z) = masked_phase_data.shape

    for i in range(z):
        # 选择哪个方向的切片都可以
        silce = masked_phase_data[:, :, i]
        silce = np.rot90(silce, k=-1)  # 逆时针旋转 90 度
        # 保存图像
        # 归一化切片数据到0-255范围，并转换为uint8类型
        slice_data = ((silce - np.min(global_min)) / (np.max(global_max) - np.min(global_min)) * 255).astype(np.uint8)

        # 将当前切片保存为 PNG
        # output_file = os.path.join(output_dir, '{}.png'.format(i))
        
        # 水平翻转
        slice_data = np.fliplr(slice_data)
        
        if os.path.exists(output_dir)==False:
            os.makedirs(output_dir)
        
        #这里是手动设置了phase图像序列号统一为00500
        output_file=os.path.join(output_dir,'img-00500-{0:05d}.png'.format(i+1))  #要保证从1开始进行编号
        imageio.imwrite(output_file, slice_data)  
        


def patient_build_nii(patient_dict,model):
    all_data=[]
    count=0  #表示一共有多少层图像
    
    #每个病人预测
    for patient,imgs in patient_dict.items():
        # 第一阶段的训练数据
        # /mnt/hdd1/zhulu/hospital/Patient-0000808485/HD-BET/SWI
        
        # patient="Patient-"+patient
        # imageSwi_dir=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/Series-Ax SWAN new"
        # imagePhase_dir=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/Series-FILT_PHA_ Ax SWAN new"

        imageSwi_dir=f"{png_dir}/{patient}/swi"
        imagePhase_dir=f"{png_dir}/{patient}/pha"
        # imagePhase_dir = f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient}/Series-FILT_PHA_ Ax SWAN new"
       
        
        #确保数据完整性无误
        if not os.path.exists(imageSwi_dir) or not os.path.exists(imagePhase_dir): # 因为使用了第二阶段的数据，但是第一阶段的原始数据目录下没有第二阶段文件，所以排除一下
            continue
        
        Swi_imgs_len=len(os.listdir(imageSwi_dir))
        Phase_imgs_len=len(os.listdir(imagePhase_dir))
        if Phase_imgs_len==0:
            print("empty: "+ imagePhase_dir)
            continue
            # return
        if Swi_imgs_len!=Phase_imgs_len: # 在新的数据中两个文件夹下不会有其他的内容,导致数量不一致
            print("error: "+ imageSwi_dir)
            continue
            # return

        depth=16
        # img_prefix="-".join(imgs[0].split("-")[:2])
        #针对于每个层面都要进行出血点的检测和3D nii文件的构造
        for index,img in enumerate(imgs):  #此处是一个病人存在标注的所有层面，此时每一张图像都要进行预测
            count+=1
            
            image_swi=cv2.imread(os.path.join(imageSwi_dir,img))
            image_swi_draw=image_swi.copy()  #进行图像绘制，在image_swi_draw上的操作不会影响image_Swi
            height,width=image_swi.shape[0],image_swi.shape[1]  #按照矩阵来读尺寸（行，列）
            # if height!=512 or width!=512:
            #     print("error: ",img,image_swi.shape)
                
            
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
                    # cv2.rectangle(image_swi_draw, (int(xyxy[0, 0]), int(xyxy[0, 1])), (int(xyxy[0, 2]), int(xyxy[0, 3])), color=(255,0,0), thickness=1)
                    # cv2.imwrite(os.path.join("temp","draw-"+img),image_swi_draw)
                    #没有数据归一化操作
                    crop_Data=np.hstack((crop_swi,crop_phase))  #看一下是否是在第二个维度上进行的拼接
                    file_name=f"suspect/{patient}-{layer_num}-{j}"
                    all_data.append([crop_Data,1,file_name,xyxy.squeeze()])  #在真实测试时，是没有标签的，随便给一个真实标签，让其预测分类

    
    transform = transforms.Compose([
    transforms.ToTensor(),  #操作会进行数据归一化
    transforms.Lambda(lambda x: x.permute(1, 2, 0))  
])

    device = torch.device('cpu')
    
    val_dataset=swi_NiiDataset2(all_data,transform) #加入归一化
    valid_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    swi_model_3D.load_state_dict(torch.load(f"parameters/res_model/swi/CMB_3DCNN_lr0001_90_adamw_focal90_gamma2_none_last.pth",map_location=device))
    pha_model_3D.load_state_dict(torch.load(f"parameters/res_model/phase/CMB_3DCNN_norm_lr0001_adamw_focal90_last.pth",map_location=device))
    # model_3D.load_state_dict(torch.load(f"parameters/{resolution}/CMB_3DCNN_new4.pth"))
    # swi预测为出血点的解雇哦
    pha_data=predict_model(swi_model_3D,valid_loader) 
    
    # valid_files, val_lbl=read_swi_data(file_path)
    valid_dataset = phase_NiiDataset(pha_data,transform)  #原本的img没有transform
    # valid_dataset = phase_NiiDataset(pha_data,transform)  #原本的img没有transform，因此在这里还需要
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
    #pha预测为出血点的结果
    phase_val_model(pha_model_3D, valid_loader)
    
    

if __name__ == "__main__":
    args=parse_args()
    dicom_dir=args.dicom_dir
    nii_dir=args.nii_dir
    png_dir=args.png_dir
    
    fixed_width = 16  # 设置固定宽度
    fixed_height = 16 # 设置固定高度
    
    
    # 3D 分类模型加载
    swi_model_3D=swi_CNN3D(2).cpu()
    pha_model_3D=phase_CNN3D(2).cpu()

    gain,pad,shape=1.02,10,(512,512,3)  #原医学图像图只有一个通道，是灰白图
    BGR=True
    reshape_size=16
    #dicom转nii
    for dicom_patient in os.listdir(dicom_dir):
        if "0018847234" !=dicom_patient:
            continue
        dicom_patient_dir=os.path.join(dicom_dir,dicom_patient)
        for dicom_data in os.listdir(dicom_patient_dir):  #在一个病人下有两个dicom文件，一个是swi，一个是pha
            dcm_p=os.path.join(dicom_patient_dir,dicom_data)
            original_image = load_dicom(dcm_p)
            norm_image=original_image
            
            patient_nii = Path(dicom_patient_dir).stem
            suffix="swi.nii" if "LocalP" not in dicom_data else "pha.nii"
            
            if not os.path.exists(os.path.join(nii_dir,patient_nii)):
                os.makedirs(os.path.join(nii_dir,patient_nii))
            
            sitk.WriteImage(norm_image, os.path.join(nii_dir,patient_nii, suffix))
        
    #去头骨项目，转PNG格式
    for nii_patient in os.listdir(nii_dir):
        swi_nii_file=os.path.join(nii_dir,nii_patient,"swi.nii")
        pha_nii_file=os.path.join(nii_dir,nii_patient,"pha.nii")
        # if os.path.exists(swi_nii_file) and os.path.exists(pha_nii_file):
        #     continue
        
        # if not os.path.exists(swi_nii_file) and os.path.exists(pha_nii_file):
        #     continue
        
        command = f"hd-bet -i {swi_nii_file} -device cpu -tta 0" #在相同路径下输出bet,以及mask文件
        
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Processed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error processing : {e}")

        prefix = Path(swi_nii_file).stem
        prefix=os.path.join(nii_dir,nii_patient,prefix)
        swi_mask=prefix+"_bet_mask.nii.gz"
        swi_bet=prefix+"_bet.nii.gz"
        pha_output_dir=os.path.join(png_dir,nii_patient,"pha")
        swi_output_dir=os.path.join(png_dir,nii_patient,"swi")
        
        # #将nii转png
        phase_mask_extract(swi_mask,pha_nii_file,pha_output_dir) #将pha nii提取成png
        nii2img(swi_bet,swi_output_dir) #将swi nii提取成png
        
    
    """ 所有层面图像 """
    #先测一个病人,以病人为单位预测
    patient_dict=dict()
    
    val_patients=["Patient-0018699418","Patient-0018847234","Patient-0019014525","Patient-0019639742"]

    
    patient_dirs=os.path.join(png_dir)
    for patient_dir in os.listdir(patient_dirs):
        
        patient_img_path=os.path.join(patient_dirs,patient_dir,"swi")
        patient_imgs=[img for img in os.listdir(patient_img_path) if img.endswith(".png")]
        patient_imgs=sort_files(patient_imgs) #根据图像的切片号进行排序，后面才能直接根据index进行上下层读取
        patient_dict[patient_dir]=patient_imgs  #切面没有进行排序
    
    from ultralytics import YOLO
    from ultralytics.utils.ops import xyxy2xywh,clip_boxes,xywh2xyxy
    # 加载YOLOv8模型
    model = YOLO("runs/detect/train34/weights/best.pt").to("cpu")   # 模型位置
    patient_build_nii(patient_dict,model)

    process_dir("txt/patient")
