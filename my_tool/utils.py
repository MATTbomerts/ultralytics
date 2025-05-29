import os 
import re
import cv2
import numpy as np
import glob
import sys


def get_patient_data(patient_dir):
    label_dir=patient_dir.replace("blood_anno4","DATA3")

    swi="swi"
    # swi="Series-Ax SWAN new"
    pha="phase"
    # pha="Series-FILT_PHA_ Ax SWAN new"
    
    patient_swi_dir=os.path.join(patient_dir,swi)
    #读出的还是原本的图像名称，没有病人前缀
    swi_imgs=[img for img in os.listdir(patient_swi_dir) if img.endswith(".png")]  
    patient_label_dir=os.path.join(label_dir,"label")
    # labels=os.listdir(patient_label_dir)
    # patient_phase_dir=os.path.join(patient_dir,pha)
    patient_phase_dir=os.path.join(patient_dir,pha)
    phase_imgs=[img for img in os.listdir(patient_phase_dir) if img.endswith(".png")]
    
    # return swi_imgs,labels,phase_imgs
    return swi_imgs,phase_imgs

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
        

def img_stack(imageSwi_dir,imagePhase_dir,swi_imgs,pha_imgs,index,second_flag):
    """_summary_:将16层的swi和phase图像进行堆叠

    Args:
        imageSwi_dir (_type_): swi文件夹路径（读取图像文件时需要）
        imagePhase_dir (_type_): phase文件夹路径（读取图像文件时需要）
        swi_imgs (_type_): swi图像文件名列表
        pha_imgs (_type_): phase图像文件名列表
        index (_type_): 当前图像在所有图像中的位置（-1关系，从0开始）
        second_flag (_type_): 另外一种设备标识符

    Returns:
        _type_: _description_
    """
    depth=16
    if len(swi_imgs)!=len(pha_imgs):
        print("error!") 
    image_swis=[]
    image_phases=[]
    phase_sequence=os.listdir(imagePhase_dir)[0].split("-")[1]
    if index-7>=0 and index+8<=len(swi_imgs)-1:#index就是该图像在所有图像中的按层数排序的位置
        for i in range(depth):  #验证没有问题
            Simage_depth_path=swi_imgs[index-7+i]  
            image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
            image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
            image_swis.append(image_Sdepth_data)
            
            deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
            img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
            image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
            if second_flag:  #进行不同设备的统一操作
                image_Pdepth_data=255-image_Pdepth_data
            image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
            image_phases.append(image_Pdepth_data)
            
    else:
        if index-7<0: #当前层以及前面层不足8，后面层多取
            for i in range(depth): 
                Simage_depth_path=swi_imgs[i]  # 直接从前向后遍历16层
                image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                image_swis.append(image_Sdepth_data)
                
                #取出所有层数，去找对应的相位图
                deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                
                if second_flag: #将第一层相位图取反，接下来拼接层也进行取反
                    image_Pdepth_data=255-image_Pdepth_data
                
                image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                image_phases.append(image_Pdepth_data)
    
        elif index+8>len(swi_imgs)-1: #后面层不足8，前面层多取，直接从后向前定位16层
            start=len(swi_imgs)-1-15
            for i in range(start,len(swi_imgs)): # 表示从start到len(imgs_path)-1
                Simage_depth_path=swi_imgs[i]  # 直接从前向后遍历16层
                image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                image_swis.append(image_Sdepth_data)
                
                deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                
                if second_flag: #将第一层相位图取反，接下来拼接层也进行取反
                    image_Pdepth_data=255-image_Pdepth_data
                
                image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                image_phases.append(image_Pdepth_data)
    
    image_swi=np.stack(image_swis,axis=2)  #现在得到的是完整的图像，没有经过裁剪   
    image_phase=np.stack(image_phases,axis=2)  #现在得到的是完整的图像，没有经过裁剪  
    if (image_swi.shape[2]!=16):
        print("error depth :",imageSwi_dir,image_swi.shape)
    if (image_phase.shape!=image_swi.shape):
        print("error shape:",imagePhase_dir,image_phase.shape)
        
    return image_swi,image_phase

# 数据加载函数
def read_data(class_names, class_labels,resolution):
    """_summary_:从整个数据集中读取训练集和测试集数据

    Args:
        class_names (_type_): _description_：CMB 和 Non-CMB
        class_labels (_type_): _description_：两个类别对应的标签
        resolution (_type_): _description_：高低分辨率分开

    Returns:
        _type_: _description_：读到的所有标签的数据
    """
     #class_names:[CMB,Non-CMB] ,class_labels:[1,0]
    data_fold=[]
    label=[]
    # 从不同的文件夹读数据时，文件夹名就是标签label
    for pos, sel in enumerate(class_names): # 将CMB和CMB一起读出来，但是先读一种，再另一种，在dataloader中进行打乱
        
        datas=sorted(glob.glob(f"/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/{resolution}/all/{sel}/*.nii"))
        for data in datas:
            data_fold.append(data)
            label.append(class_labels[pos])
    return data_fold,label

def read_swi_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_fold=[]
        label=[]
        file_dir1="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all"
        file_dir2="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add3"
        for line in lines:
            # 按 | 分割数据
            parts = line.strip().split('|')
            data_path, true_label, pred_label, bounding_box,logits = parts
            if pred_label.strip()=="1": #将swi模型预测为CMB的数据读出来，经过phase再判断
                if os.path.exists(os.path.join(file_dir1,data_path)) and os.path.exists(os.path.join(file_dir2,data_path)):
                    print(data_path)
                    sys.exit()
                if os.path.exists(os.path.join(file_dir1,data_path)):
                    file_dir=file_dir1
                elif os.path.exists(os.path.join(file_dir2,data_path)):
                    file_dir=file_dir2
                data_fold.append(os.path.join(file_dir,data_path))
                label.append(int(true_label.strip()))  
                
    return data_fold,label      

# 数据加载函数
def read_add_data(class_names, class_labels,file_dir,resolution):
    """_summary_:从整个数据集中读取训练集和测试集数据

    Args:
        class_names (_type_): _description_：CMB 和 Non-CMB
        class_labels (_type_): _description_：两个类别对应的标签
        resolution (_type_): _description_：高低分辨率分开

    Returns:
        _type_: _description_：读到的所有标签的数据
    """
     #class_names:[CMB,Non-CMB] ,class_labels:[1,0]
    data_fold=[]
    label=[]
    # 从不同的文件夹读数据时，文件夹名就是标签label
    for pos, sel in enumerate(class_names): # 将CMB和CMB一起读出来，但是先读一种，再另一种，在dataloader中进行打乱
        # datas=sorted(glob.glob(f"/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add/{sel}/*.nii"))
        datas=sorted(glob.glob(f"{file_dir}/{sel}/*.nii"))
        for data in datas:
            data_fold.append(data)
            label.append(class_labels[pos])
    return data_fold,label