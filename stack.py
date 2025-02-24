import os
import cv2
import ast

# file_dir=["/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all","/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add3"]
file_dir="/mnt/hdd1/zhulu/blood_stage2/tmp/total"
swi=["swi","Series-Ax SWAN new"]
phase=["phase","Series-FILT_PHA_ Ax SWAN new"]



for patient in os.listdir(file_dir):
    swi_name=swi[0]
    phase_name=phase[0]
    if swi[0] not in os.listdir(os.path.join(file_dir,patient)):
        swi_name=swi[1]
        phase_name=phase[1]
    swi_seq=os.listdir(os.path.join(file_dir,patient,swi_name))[0].split("-")[1]
    phase_seq=os.listdir(os.path.join(file_dir,patient,phase_name))[0].split("-")[1]
    swi_dir=os.path.join(file_dir,patient,swi_name)
    phase_dir=os.path.join(file_dir,patient,phase_name)
    stack_file=os.path.join(file_dir,patient,"stack")
    if os.path.exists(stack_file)==False:
        os.makedirs(stack_file)
    swi_imgs=[x for x in os.listdir(swi_dir) if x.endswith(".png")]
    for swi_img_file in swi_imgs:
        layer_num=swi_img_file.split("-")[-1].split(".")[0]
        swi_img=cv2.imread(os.path.join(swi_dir,swi_img_file))
        phase_sequence=os.listdir(phase_dir)[0].split("-")[1]
        img_phase_path="-".join(["img",phase_sequence,layer_num])+".png"
        image_Phase_data=cv2.imread(os.path.join(phase_dir,img_phase_path)) 
        combined_img=cv2.hconcat([swi_img,image_Phase_data])
        cv2.imwrite(os.path.join(stack_file,swi_img_file),combined_img)
        
        
        
        
    