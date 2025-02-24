import os
import cv2
import ast
import numpy as np
import nibabel as nib
import torch
from ultralytics.utils.metrics import box_iou


def pred2nii(patient_dirs,patient_dir):
    non2CMB=0
    wrong_img_pred_dict=dict()
    right_predict_dict=dict()
    
    with open("filtered.txt","r") as f:  #这个是出血点为单位，画图应该以图像为单位
    # with open(f"secondStage/filtered/{patient_dir}.txt","r") as f:  #这个是出血点为单位，画图应该以图像为单位
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
                
                # if len(wrong_img_pred_dict[layer_num])>1:
                #     print(wrong_img_pred_dict[layer_num])
                    # print(f"layer_num: {layer_num},wrong_img_pred_dict: {wrong_img_pred_dict[layer_num]}")
                #img_name：Patient-0000808485-00100-1 包含了这一层面的第几个预测框
                
                
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

    all_layers=[]
    false_pred_box=0
    
    swi_dir="Series-Ax SWAN new"
    phase_dir="Series-FILT_PHA_ Ax SWAN new"
    
    pred_file=os.path.join(patient_dirs,patient_dir)
    
    swi_imgs= [f for f in os.listdir(os.path.join(pred_file,swi_dir)) if f.endswith('.png')]
    # 对图像按层面数进行排序
    swi_imgs = sorted(swi_imgs, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    
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
                x1,y1,x2,y2=box[0]
                cv2.rectangle(image_swi, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
                cv2.rectangle(image_phase, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
        
        if layer_num in right_predict_dict:  # 再正式预测时不会出现正确预测的数据集
            for box in right_predict_dict[layer_num]:
                box = ast.literal_eval(box)
                x1,y1,x2,y2=box[0]
                cv2.rectangle(image_swi, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
                cv2.rectangle(image_phase, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2) # 绿色
                # 添加标记文本
                label = "11"  # 你想要显示的标签，可以根据需要进行更改
                text_position = (int(x1), int(y1) - 10)  # 在矩形框上方的位置
                cv2.putText(image_swi, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
                cv2.putText(image_phase, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
                
                
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

def filter(pred_txt):
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
                        if pred==pred2:  
                            filtered_index.append(curr_index)  #如果存在重合则将下一层加入到去重中
                            if real2=="1" and real=="0":  #要删除下一行,因此需要修改当前行
                                # parts=line.strip().split("|")
                                # parts[1]=parts[1].replace("0","1")
                                # line="|".join(parts)
                                line=line.replace("|0","|1")
                        #region        
                        # if real==real2 and pred==pred2:  #如果两个框的预测和标签一样，并且存在重合
                        #     filtered_index.append(curr_index)  #如果存在重合则加入到去重中
                        # if  real=="0" and real2=="1": #删除当前层,并且下一层为出血点
                        #     filtered_index.append(index)
                        # if real=="1" and real2=="0": #如果当前层的标注是1,但是下一层没有标注了,则下一层标注又变为0
                        #     filtered_index.append(curr_index) 
                        #endregion
                    else:
                        pass
            if index not in filtered_index:  #并且要保证都是同一个病人
                filtered_lines.append(line)            
    # 将filtered_lines写入到新的文件中
    output_file = "filtered.txt"
    with open(output_file, 'w') as out_f:
        for filtered_line in filtered_lines:
            out_f.write(filtered_line)