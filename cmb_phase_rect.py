import os
import cv2
import ast
import nibabel as nib
from tqdm import tqdm

# CMBs_dir="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all/CMB"
CMBs_dir="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all/Non_CMB"


# file_dir=["/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all","/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add3"]
file_dir="/mnt/hdd1/zhulu/blood_stage2/tmp/total"
swi=["swi","Series-Ax SWAN new"]
phase=["phase","Series-FILT_PHA_ Ax SWAN new"]

wrong_patient=set()
patient_files = {}

for cmb in os.listdir(CMBs_dir):
    patient_name="-".join(cmb.split("-")[:2])
    if patient_name not in patient_files:
        patient_files[patient_name] = []
    patient_files[patient_name].append(cmb)
    
# for patient,files in patient_files.items():
for patient, files in tqdm(patient_files.items(), desc="Processing patients", unit="patient"):
    wrong_patient.add(patient)
    for file in files:
        nii_data = nib.load(os.path.join(CMBs_dir,file))
        img = nii_data.get_fdata()
        descrip = nii_data.header['descrip'].item().decode('utf-8')
        bbox_str = descrip.split('Bounding boxes: ')[1]  # 提取列表部分
        box = ast.literal_eval(bbox_str) #将字符串列表转换为数字列表
        
        patient_name="-".join(file.split("-")[:2])
        layer_name=file.split("-")[-2]
        
        phase_name=phase[0]
        swi_name=swi[0]
        
        if swi[0] not in os.listdir(os.path.join(file_dir,patient_name)):
            swi_name=swi[1]
            phase_name=phase[1]
        
        swi_seq=os.listdir(os.path.join(file_dir,patient_name,swi_name))[0].split("-")[1]
        swi_img_path=os.path.join(file_dir,patient_name,swi_name,f"img-{swi_seq}-{layer_name}.png")
        phase_seq=os.listdir(os.path.join(file_dir,patient_name,phase_name))[0].split("-")[1]
        phase_img_path=os.path.join(file_dir,patient_name,phase_name,f"img-{phase_seq}-{layer_name}.png")
        swi_img=cv2.imread(swi_img_path)
        
        
        phase_img=cv2.imread(phase_img_path)
        # 是这一层的第几个标注
        count_info=file.split("-")[-1].split(".")[0]

        # 在 swi_img 上标记框信息
        text_position_swi = (int(box[0]), int(box[1]) - 10)  # 在框的左上角上方显示
    
        # 在 phase_img 上标记框信息
        text_position_phase = (int(box[0]), int(box[1]) - 10)  # 在框的左上角上方显示
        cv2.putText(phase_img, count_info, text_position_phase, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2) #黄色（0,255,255）
        cv2.rectangle(phase_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,255),1)
        cv2.rectangle(swi_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,255),1) #红色（0,0,255）表示原本非出血预测为出血
        
        # cv2.imwrite("temp/phase.png", phase_img)
        # cv2.imwrite("temp/swi.png", swi_img)

        cv2.imwrite(phase_img_path, phase_img)
        cv2.imwrite(swi_img_path, swi_img)
                    
with open("wrong_patient.txt","w") as f:
    for patient in wrong_patient:
        f.write(patient+"\n")