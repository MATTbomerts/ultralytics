import os
import cv2
import ast
phase_pred="secondStage/phase_pred/test_all_samples_predict.txt"
# file_dir=["/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/all","/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add3"]
file_dir="/mnt/hdd1/zhulu/blood_stage2/tmp/total"
swi=["swi","Series-Ax SWAN new"]
phase=["phase","Series-FILT_PHA_ Ax SWAN new"]

wrong_patient=set()

with open(phase_pred,"r") as f:
    lines=f.readlines()
    for line in lines:
        file_path,label,predict,box=line.strip().split("|")
        box = ast.literal_eval(box) #将字符串列表转换为数字列表
        patient_name="Patient-"+file_path.split("/")[-1].split("-")[1]
        layer_name=file_path.split("/")[-1].split("-")[-2]
        
        wrong_patient.add(patient_name)
        if predict.strip()!=label.strip(): #只要预测错就显示出来
            
            swi_name=swi[0]
            phase_name=phase[0]
            if swi[0] not in os.listdir(os.path.join(file_dir,patient_name)):
                swi_name=swi[1]
                phase_name=phase[1]
            swi_seq=os.listdir(os.path.join(file_dir,patient_name,swi_name))[0].split("-")[1]
            phase_seq=os.listdir(os.path.join(file_dir,patient_name,phase_name))[0].split("-")[1]
            swi_img_path=os.path.join(file_dir,patient_name,swi_name,f"img-{swi_seq}-{layer_name}.png")
            phase_img_path=os.path.join(file_dir,patient_name,phase_name,f"img-{phase_seq}-{layer_name}.png")
            swi_img=cv2.imread(swi_img_path)
            phase_img=cv2.imread(phase_img_path)
            
            # 定义框信息的文本
            count_info=file_path.split("/")[-1].split(".")[0].split("-")[-1]

            # 在 swi_img 上标记框信息
            text_position_swi = (int(box[0]), int(box[1]) - 10)  # 在框的左上角上方显示
            cv2.putText(swi_img, count_info, text_position_swi, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 在 phase_img 上标记框信息
            text_position_phase = (int(box[0]), int(box[1]) - 10)  # 在框的左上角上方显示
            cv2.putText(phase_img, count_info, text_position_phase, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            
            if predict.strip()=="1":
                cv2.rectangle(swi_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2) #红色表示原本非出血预测为出血
                cv2.rectangle(phase_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
            else:
                cv2.rectangle(swi_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2) #绿色表示原本出血预测为非出血
                cv2.rectangle(phase_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
            
            
            # cv2.imwrite("temp/swi_img.png", swi_img)
            cv2.imwrite(swi_img_path, swi_img)
            cv2.imwrite(phase_img_path, phase_img)
            
            
            
with open("wrong_patient.txt","w") as f:
    for patient in wrong_patient:
        f.write(patient+"\n")
            
            