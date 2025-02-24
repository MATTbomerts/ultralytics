import os
import cv2
import numpy as np
import nibabel as nib
from my_tool.utils import get_patient_data,sort_files,img_stack
from secondStage.utils import rotate,flip,gaussianBlur

BGR=True
add_dir="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add1"
write_path="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/box"
#要将新标注的数据构造出nii文件，然后进行数据增强，再加入到训练数据集中
new_file_path="/mnt/hdd1/zhulu/blood_stage2/blood_anno3/PNG/PNG"
new_patients=os.listdir(new_file_path)
for patient in new_patients:
    label_dir=os.path.join(new_file_path,patient,"label")
    if os.path.exists(label_dir)==False: #如果该病人并没有标注，则跳过
        continue
    patient_dir=os.path.join(new_file_path,patient)
    swi_dir=os.path.join(patient_dir,"swi")
    phase_dir=os.path.join(patient_dir,"phase")
    swi_imgs= [f for f in os.listdir(swi_dir) if f.endswith('.png')]
    phase_imgs= [f for f in os.listdir(phase_dir) if f.endswith('.png')]
    swi_imgs=sort_files(swi_imgs)
    phase_imgs=sort_files(phase_imgs)
    swi_img=cv2.imread(os.path.join(swi_dir,swi_imgs[0]))
    height,width=swi_img.shape[0],swi_img.shape[1]
    for label in os.listdir(label_dir):
        label_file=os.path.join(label_dir,label)
        layer_num=label.split("-")[-1].split(".")[0]
        index=int(layer_num)-1
        image_swis,image_phases=img_stack(swi_dir,phase_dir,swi_imgs,phase_imgs,index,False)
        with open(label_file, 'r') as file:
            lines = file.readlines()
            for j,line in enumerate(lines): #针对于一张图像，多个标注
                #center_x表示列，center_y表示中心行
                id,center_x,center_y,width_norm,height_norm=line.strip().split()  #这里拿到的是压缩比，归一化的结果
                #先直接将box的宽高设置为16x16
                center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), 16, 16
                # center_x, center_y, w, h = round(width*float(center_x)), round(height*float(center_y)), round(float(width_norm)*width), round(float(height_norm)*height)
                x1=center_x-w//2 #x表示的列，y表示的行，从txt中的中心转换为，现在h,w用的没错
                y1=center_y-h//2
                x2=center_x+w//2
                y2=center_y+h//2
                #现在存在cmb和non-cmb两种情况
                #因为图像在内存中以height,width,channel方式存储，第一个维度对应y方向，第二个维度对应x方向
                crop_swi = image_swis[y1 : y2, x1 : x2, :: (1 if BGR else -1)]
                crop_phase=image_phases[y1 : y2, x1 : x2, :: (1 if BGR else -1)]
                
                img_swi_draw=cv2.imread(os.path.join(swi_dir,swi_imgs[index]))
                cv2.rectangle(img_swi_draw, (x1, y1), (x2, y2), color=(0,0,255), thickness=1) #蓝色
                write_dir=os.path.join(write_path,patient)
                if os.path.exists(write_dir)==False:
                    os.makedirs(write_dir)
                cv2.imwrite(os.path.join(write_dir,f"{patient}-{layer_num}-{j}.png"),img_swi_draw)
                #没有数据归一化操作
                crop_Data=np.hstack((crop_swi,crop_phase))
                if 0 not in crop_Data.shape: #如果尺寸没有问题，则创建nii文件
                    scaling = np.eye(4) 
                    nii_img = nib.Nifti1Image(crop_Data, scaling)  #crop_data的数据值不会受到scaling的影响
                    nii_img.header['descrip'] = f'Bounding boxes: {[x1,y1,x2,y2]}'  # 将坐标信息也存放到nii文件中
                if id=="0":
                    type="Non_CMB"
                else:
                    type="CMB"
                    # print("yes")
                base_name=f"Patient-{patient}-{layer_num}-{j}.nii"
                
                # nib.save(nii_img,os.path.join(add_dir,type,base_name))
                # #数据翻转
                # flip_horizontal=flip(nii_img,1) #因为直接是在原本16x16的图像上进行操作，所以坐标框不会变化
                # flip_vertical=flip(nii_img,0)
                # flip_horizontal_name=base_name.split(".")[0]+"_flip_horizontal.nii"
                # flip_vertical_name=base_name.split(".")[0]+"_flip_vertical.nii"
                # nib.save(flip_horizontal,os.path.join(add_dir,type,flip_horizontal_name))
                # nib.save(flip_vertical,os.path.join(add_dir,type,flip_vertical_name))
                
                # #数据旋转，在方法内部进行了swi和phase图像的分离
                # rotate_90=rotate(nii_img,90)   #顺时针旋转没问题，但是在ITK-SNAP中显示存在镜像问题
                # rotate_180=rotate(nii_img,180) #旋转部分都没有问题
                # rotate_270=rotate(nii_img,270)
                # rotae_90_name=base_name.split(".")[0]+"_rotae_90.nii"
                # rotate_180_name=base_name.split(".")[0]+"_rotate_180.nii"
                # rotate_270_name=base_name.split(".")[0]+"_rotate_270.nii"
                # nib.save(rotate_90,os.path.join(add_dir,type,rotae_90_name))
                # nib.save(rotate_180,os.path.join(add_dir,type,rotate_180_name))
                # nib.save(rotate_270,os.path.join(add_dir,type,rotate_270_name))
                
                # #高斯操作
                # k_size=3
                # gaussian_blur=gaussianBlur(nii_img,k_size)
                # gaussian_blur_name=base_name.split(".")[0]+"_gaussian_blur.nii"
                # nib.save(gaussian_blur,os.path.join(add_dir,type,gaussian_blur_name))
                    
                    
    