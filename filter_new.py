
from ultralytics.utils.metrics import box_iou
import ast
import torch
import cv2
import os

#目前只针对单个病人进行上述操作
pred_txt="secondStage/all_samples_predict.txt"

data_root="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
swi="Series-Ax SWAN new"
filtered_lines=[]
filtered_index=[]

with open(pred_txt, 'r') as f:
    lines = f.readlines()
    current_patient=None
    for index,line in enumerate(lines):
        if index not in filtered_index:  #并且要保证都是同一个病人
            filtered_lines.append(line)
        data_path,real,pred,box=line.strip().split("|")  #每一行就是一个box，而不是一个层面的所有box
        real=real.strip()
        pred=pred.strip()
        data_type,patient_layer=data_path.split("/")[-2:]
        patient="-".join(patient_layer.split("-")[:2])
        layer=patient_layer.split("-")[-2]
        box=ast.literal_eval(box)
        #一层有多个box，每个box向下去找相交的框
        for res_index,res_line in enumerate(lines[index+1:]):
            curr_index=index+res_index+1  #绝对索引
            data_path2,real2,pred2,box2=res_line.strip().split("|")
            real2=real2.strip()
            pred2=pred2.strip()
            data_type2,patient_layer2=data_path2.split("/")[-2:]
            patient2="-".join(patient_layer2.split("-")[:2])
            layer2=patient_layer2.split("-")[-2]
            if patient!=patient2:
                break
            if int(layer2)>int(layer)+1: 
                break
            if int(layer2)==int(layer):
                continue
            else: #只有在下一层的时候才计算iou
                box2=ast.literal_eval(box2)
                iou=box_iou(torch.tensor(box),torch.tensor(box2))  #一个一个计算，因此实际只有一个值
                iou=iou.squeeze()
                if iou>0.5:  #说明高度重合
                    #可视化
                    img_name="img-00005-"+layer+".png"
                    img_path=os.path.join(data_root,patient,swi,img_name)
                    img=cv2.imread(img_path)
                    cv2.rectangle(img,(box[0][0],box[0][1]),(box[0][2],box[0][3]),(0,0,255),1)  #红色
                    cv2.rectangle(img,(box2[0][0],box2[0][1]),(box2[0][2],box2[0][3]),(0,255,0),1) # 绿色
                    cv2.imwrite("temp/iou.png",img)
                    if real==real2 and pred==pred2:  #如果两个框的预测和标签一样，并且存在重合
                        filtered_index.append(curr_index)  #如果存在重合则加入到去重中
                else:
                    pass
                    
# 将filtered_lines写入到新的文件中
output_file = "filtered_lines.txt"
with open(output_file, 'w') as out_f:
    for filtered_line in filtered_lines:
        out_f.write(filtered_line)

                    