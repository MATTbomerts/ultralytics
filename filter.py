
from ultralytics.utils.metrics import box_iou
import ast
import torch
import cv2
import os

pred_dir="secondStage/pred"
for pred_file in os.listdir(pred_dir):
    pred_txt=os.path.join(pred_dir,pred_file)
    #目前只针对单个病人进行上述操作
    # pred_txt="secondStage/all_samples_predict.txt"

    patient_name=pred_file.split("_")[-1].split(".")[0]
    # data_root="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
    swi="swi"
    filtered_lines=[]
    filtered_index=[]

    with open(pred_txt, 'r') as f:
        lines = f.readlines()
        current_patient=None
        for index,line in enumerate(lines):
            data_path,real,pred,box=line.strip().split("|")  #每一行就是一个box，而不是一个层面的所有box
            real=real.strip()
            pred=pred.strip()
            data_type,patient_layer=data_path.split("/")[-2:]
            patient="-".join(patient_layer.split("-")[:1])
            layer=patient_layer.split("-")[-2]
            box=ast.literal_eval(box)
            #一层有多个box，每个box向下去找相交的框
            for res_index,res_line in enumerate(lines[index+1:]):
                curr_index=index+res_index+1  #绝对索引
                data_path2,real2,pred2,box2=res_line.strip().split("|")
                real2=real2.strip()
                pred2=pred2.strip()
                data_type2,patient_layer2=data_path2.split("/")[-2:]
                patient2="-".join(patient_layer2.split("-")[:1])
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
                        #region
                        # #可视化
                        # img_name="img-00005-"+layer+".png"
                        # img_path=os.path.join(data_root,patient,swi,img_name)
                        # img=cv2.imread(img_path)
                        # cv2.rectangle(img,(box[0][0],box[0][1]),(box[0][2],box[0][3]),(0,0,255),1)  #红色
                        # cv2.rectangle(img,(box2[0][0],box2[0][1]),(box2[0][2],box2[0][3]),(0,255,0),1) # 绿色
                        # cv2.imwrite("temp/iou.png",img)
                        #如果两个框的预测结果一致，并且存在重合,但可能把标签也去掉了,前一层其实没有人工标签,但是下一层有人工标签,导致最终预测中看似少预测到一些
                        #在测量指标时,可以根据人工标注来判断一下
                        #endregion
                        if pred==pred2:   #如果预测都相同
                            filtered_index.append(curr_index)  #如果存在重合则将下一层加入到去重中
                            if real2=="1" and real=="0":  #要删除下一行,因此需要修改当前行
                                parts=line.strip().split("|")
                                parts[1]=parts[1].replace("0","1")
                                line="|".join(parts)
                                line=line+"\n"
                                # line=line.replace("|0","|1") #会把标签以及预测都改成1，就错了
                        #region        
                        # if real==real2 and pred==pred2:  #如果两个框的预测和标签一样，并且存在重合
                        #     filtered_index.append(curr_index)  #如果存在重合则加入到去重中
                        # if  real=="0" and real2=="1": #删除当前层,并且下一层为出血点
                        #     filtered_index.append(index)
                        # if real=="1" and real2=="0": #如果当前层的标注是1,但是下一层没有标注了,则下一层标注又变为0
                        #     filtered_index.append(curr_index) 
                        #endregion
                        if real==real2: #直接如果连续层标签都一样并且IOU比较接近就删除
                            #如果连续多层里面有一个识别对了就算对，
                            pass
                    else:
                        pass
            if index not in filtered_index:  #并且要保证都是同一个病人
                filtered_lines.append(line)            
    # 将filtered_lines写入到新的文件中
    output_file = f"secondStage/filtered/{patient_name}.txt"
    with open(output_file, 'w') as out_f:
        for filtered_line in filtered_lines:
            out_f.write(filtered_line)

                    