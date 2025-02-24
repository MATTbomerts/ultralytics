import os
import cv2
import ast
import numpy as np
import nibabel as nib

except_patient=["0002629740"]
patient_dirs="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
# val_patients=["Patient-0018699418","Patient-0018847234","Patient-0019014525","Patient-0019639742"]
val_patients=["Patient-0000032463","Patient-0000060881","Patient-0000137786","Patient-0000284759"]
for patient_dir in os.listdir(patient_dirs):
    if patient_dir in except_patient:
        continue
    if patient_dir not in val_patients:
        continue
    pred_file=os.path.join(patient_dirs,patient_dir)
    # pred_file="/mnt/hdd1/zhulu/blood_stage2/blood_anno2/PNG/0001232716/" #对该文件夹中的图像进行绘制并写到新的文件夹中
    # pred_file="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/Patient-0019014525" #对该文件夹中的图像进行绘制并写到新的文件夹中
    swi_dir="Series-Ax SWAN new"
    phase_dir="Series-FILT_PHA_ Ax SWAN new"
    swi_imgs= [f for f in os.listdir(os.path.join(pred_file,swi_dir)) if f.endswith('.png')]
    # 对图像按层面数进行排序
    swi_imgs = sorted(swi_imgs, key=lambda x: int(x.split('-')[-1].split('.')[0]))

    non2CMB=0
    wrong_img_pred_dict=dict()
    right_predict_dict=dict()
    false_cmb_dict=dict()

    # with open("secondStage/all_samples_predict_Patient.txt","r") as f:  #这个是出血点为单位，画图应该以图像为单位
    # with open(f"secondStage/shape/{patient_dir}.txt_shape.txt","r") as f:  #这个是出血点为单位，画图应该以图像为单位
    with open(f"secondStage/phase_pred/all_samples_predict_{patient_dir}.txt","r") as f:  #这个是出血点为单位，画图应该以图像为单位
        lines=f.readlines()
        for line in lines:
            label,predict,box=line.strip().split("|")[-3:]
            # 将每一层切面正确预测和错误预测的保存起来，组织成 image_layer:wrong_box; image_layer:right_box，两个字典
            if predict.strip()=="1" and label.strip()=="0":  #将非出血点预测为出血点
                non2CMB+=1
                img_cmb_name=line.split("|")[0].split("/")[-1]
                img_name="-".join(img_cmb_name.split("-")[:-1])+".png"  #
                layer_num=img_name.split("-")[-1].split(".")[0]
                if layer_num not in wrong_img_pred_dict:
                    
                    wrong_img_pred_dict[layer_num]=[]
                    wrong_img_pred_dict[layer_num].append(box)
                else:
                    wrong_img_pred_dict[layer_num].append(box)  #将一个图像层面的所有预测框都保存下来
                
                
            if predict.strip()=="1" and label.strip()=="1":  #正确预测,在真实预测时不会出现原本label标注为1
                img_cmb_name=line.split("|")[0].split("/")[-1]
                img_name="-".join(img_cmb_name.split("-")[:-1])+".png"  #
                layer_num=img_name.split("-")[-1].split(".")[0]
                if layer_num not in right_predict_dict:
                    
                    right_predict_dict[layer_num]=[]
                    right_predict_dict[layer_num].append(box)
                else:
                    right_predict_dict[layer_num].append(box)  #将一个图像层面的所有预测框都保存下来
                
                if len(right_predict_dict[layer_num])>1:
                    print(right_predict_dict[layer_num])
            
            if predict.strip()=="0" and label.strip()=="1":
                img_cmb_name=line.split("|")[0].split("/")[-1]
                img_name="-".join(img_cmb_name.split("-")[:-1])+".png"  #
                layer_num=img_name.split("-")[-1].split(".")[0]
                if layer_num not in false_cmb_dict:
                    
                    false_cmb_dict[layer_num]=[]
                    false_cmb_dict[layer_num].append(box)
                else:
                    false_cmb_dict[layer_num].append(box)  #将一个图像层面的所有预测框都保存下来
                
                if len(false_cmb_dict[layer_num])>1:
                    print(false_cmb_dict[layer_num])
                

    all_layers=[]
    false_pred_box=0
    phase_seq=os.listdir(os.path.join(pred_file,phase_dir))[0].split("-")[1]  #得到相位图中的序列号，根据这个号后续进行拼接
    for swi_img in swi_imgs: # 要将所有的图层stack起来变成nii格式文件
        image_swi=cv2.imread(os.path.join(pred_file,swi_dir,swi_img),cv2.IMREAD_GRAYSCALE)  # 读取swi图像
        phase_img_name="-".join(["img",phase_seq,swi_img.split("-")[-1].split(".")[0]])+".png"
        image_phase=cv2.imread(os.path.join(pred_file,phase_dir,phase_img_name),cv2.IMREAD_GRAYSCALE) # 读取phase图像
        layer_num=swi_img.split("-")[-1].split(".")[0]
        if layer_num in wrong_img_pred_dict:
            for box in wrong_img_pred_dict[layer_num]:  #box得到的是一个字符串，需要转换为列表
                false_pred_box+=1
                box = ast.literal_eval(box) #将字符串列表转换为数字列表
                # x1,y1,x2,y2=box[0]
                x1,y1,x2,y2=box
                cv2.rectangle(image_swi, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
                cv2.rectangle(image_phase, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
        
        if layer_num in right_predict_dict:  # 再正式预测时不会出现正确预测的数据集
            for box in right_predict_dict[layer_num]:
                box = ast.literal_eval(box)
                # x1,y1,x2,y2=box[0]
                x1,y1,x2,y2=box
                cv2.rectangle(image_swi, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
                cv2.rectangle(image_phase, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
                # 添加标记文本
                label = "11"  # 你想要显示的标签，可以根据需要进行更改
                text_position = (int(x1), int(y1) - 10)  # 在矩形框上方的位置
                cv2.putText(image_swi, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
                cv2.putText(image_phase, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
                
        if layer_num in false_cmb_dict:  # 再正式预测时不会出现正确预测的数据集
            for box in false_cmb_dict[layer_num]:
                box = ast.literal_eval(box)
                # x1,y1,x2,y2=box[0]
                x1,y1,x2,y2=box
                cv2.rectangle(image_swi, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,255,255), thickness=2) # 白色
                cv2.rectangle(image_phase, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,255,255), thickness=2) # 白色
                
                
        joint_img=cv2.hconcat([image_swi,image_phase])   #得到的是png图像，有三个通道   
        # cv2.imwrite("temp/temp.png", joint_img)
        all_layers.append(joint_img)
        
    # print(f"false_pred_box: {false_pred_box},{non2CMB}")

    all_layers_img=np.stack(all_layers,axis=-1)  #将所有的层面图像进行stack，得到一个三维的nii图像
    scaling = np.eye(4)

    voxel_sizes = [512, 1024, 116]
    # affine = np.diag(voxel_sizes + [1])
    nii_img = nib.Nifti1Image(all_layers_img, scaling)  #crop_data的数据值不会受到scaling的影响
    # nii_save_path="/mnt/hdd1/zhulu/blood_stage2/blood_anno/predict_Result/Patient-0000808485"
    output_dir=f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient_dir}/pred"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nib.save(nii_img,f'{output_dir}/result_filtered.nii')    


# #根据id来进行训练和测试的不同的保存
# if id==str(0): #只有0号才是出血点，其余的都是干扰点，划分正样例和负样例
#     # j 是txt文件中第几行出现的标注框
#     nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/train_model/all/CMB/{patients}-{layer_num}-{j}.nii')    


    
            
# for img_name in wrong_img_pred_dict:  
#     image_swi=cv2.imread(os.path.join(pred_file,swi_dir,img_name))  # 读取swi图像
#     image_phase=cv2.imread(os.path.join(pred_file,phase_dir,img_name)) # 读取phase图像
#     for box in img_pred_dict[img_name]: #将同一个层面所有预测错误的框都绘制出来
#         x1,y1,x2,y2=box.split()
#         cv2.rectangle(image_swi, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=1) # 绿色
#         cv2.rectangle(image_phase, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=1) # 绿色

#     cv2.imwrite(os.path.join(pred_file,img_name),image_swi)
            