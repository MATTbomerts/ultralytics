import os
import shutil
data_path="/mnt/hdd1/zhulu/blood_stage2/DATA3/PNG"
label_name="label"
train_label_path="/mnt/hdd1/zhulu/hospital/labels/high_resolution/train"
val_label_path="/mnt/hdd1/zhulu/hospital/labels/high_resolution/val"

train_img_path="/mnt/hdd1/zhulu/hospital/images/high_resolution/train"
val_img_path="/mnt/hdd1/zhulu/hospital/images/high_resolution/val"

patients=os.listdir(data_path)
num_lines=0
train_add_num=0
flag=0
for patient in patients:
    patient_dir=os.path.join(data_path,patient)
    labels_path=os.path.join(patient_dir,label_name)
    if num_lines>=90: #标记位修改，从90开始按照病例放在验证集中
        flag=1
    for label_file in os.listdir(labels_path): # 每个病人的label文件，将每个病人遍历完
        label_path=os.path.join(labels_path,label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            label_file_name=f"Patient-{patient}_{label_file}"
            img_new_file_name=f"Patient-{patient}_{label_file.split('.')[0]}.png"
            img_file_path=f"{data_path}/{patient}/swi/{label_file.split('.')[0]}.png"
            
            if flag==1: #放在验证集中
                shutil.copy(label_path, os.path.join(val_label_path, label_file_name))
                shutil.copy(img_file_path, os.path.join(val_img_path, img_new_file_name))
                
            else: #放在训练集中
                shutil.copy(label_path, os.path.join(train_label_path, label_file_name))
                shutil.copy(img_file_path, os.path.join(train_img_path, img_new_file_name))
                train_add_num+=1
        num_lines+=len(lines)         
        
print(f"train_add_num:{train_add_num}")