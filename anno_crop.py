import os
import re
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm
from PIL import Image

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

hospital_dir = "/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
dirs=os.listdir(hospital_dir)

patients_dir=[dir for dir in dirs]  
second_devices=["Patient-0000406857","Patient-0000589446",'Patient-0000745885','Patient-0003537968'
                ,'Patient-0010134951','Patient-0018227242','Patient-0018463963']

flag=[] # 用来检查图像尺寸不是512的，是不是上面的second_device这一批
def niiData_build(patient_dir,train_test):
    #region
    #一个病人图像数据
    # imageSwi_dir = "/mnt/hdd1/zhulu/hospital/Patient-0019009004/Series-Ax SWAN new" 
    # imagePhase_dir = "/mnt/hdd1/zhulu/hospital/Patient-0019009004/Series-FILT_PHA_ Ax SWAN new" 
    # txt_dir = "/mnt/hdd1/zhulu/hospital/Patient-0019009004/label"
    #endregion
    
    imageSwi_dir = f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient_dir}/Series-Ax SWAN new" 
    imagePhase_dir = f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient_dir}/Series-FILT_PHA_ Ax SWAN new" 
    patients=imageSwi_dir.split("/")[-2]
    #txt还没改呢
    txt_dir = f"/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG/{patient_dir}/label"
    imgs_path = [f for f in os.listdir(imageSwi_dir) if f.endswith('.png')] #拿到这个病人的所有的图像
    imgs_path=sort_files(imgs_path) #根据图像的切片号进行排序
    
    Pimgs_path=os.listdir(imagePhase_dir)  #并不是每张图像都有标注，先查看一下数据是否存在问题
    if len(Pimgs_path)==0:
        print("empty: "+ imagePhase_dir)
    if len(imgs_path)!=len(Pimgs_path):
        print("error: "+ imageSwi_dir)
        return
    # os.makedirs(os.path.join(output_dir,patients,"swi"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir,patients,"phase"), exist_ok=True)

    depth=16 #包括当前图像的话，应该加的是15
    #貌似png文件中序列号还是没改过来
    phase_sequence=Pimgs_path[0].split("-")[1]  #得到相位图中的序列号，根据这个号后续进行拼接
    #对每一层的图像处理，看一下处理的顺序是不是从小到大.查看结果：是从小到大的顺序的
    
    
    
    for index,img_path in enumerate(imgs_path): #index就是从第0层开始的
        
        #要先判度当前layer是否有标注(是出血点)，再去读取图像进行nii文件的生成
        txt_file_name = os.path.splitext(img_path)[0]+".txt"
        if  not os.path.exists(txt_dir) or txt_file_name not in os.listdir(txt_dir):  #有可能某些层面没有标注信息
            continue
        
        #有的图像在anno中有images的信息，但是并没有产生标注
        if os.path.getsize(os.path.join(txt_dir,txt_file_name)) <= 0:
            continue
        
        layer_num=img_path.split("-")[-1].split(".")[0]  #layer_num就是样本点所在的层，不是深度的层
        #如果当前层图像有出血点标注，就读取图像数据，第二阶段的模型使用灰度图进行训练的
        image_swi=cv2.imread(os.path.join(imageSwi_dir,img_path)) #读取单张图像，不是灰度图，512，512，3
        image_swi = cv2.cvtColor(image_swi, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图

        height,width=image_swi.shape[0],image_swi.shape[1]  #对于这个病人来说下面的图像尺寸大小是固定的
        if height!=512:
            flag.append(patient_dir)
        
        #新的数据，phase图像和SWI图像的序列名没有规律，可能是00001-00009，只有层数不变固定
        #因此需要根据层数号去找对应的phase图像
        #有一个共识就是同一个文件夹下，序列号是不变的，因此拿出一张图像即可
        phase_path="-".join(["img",phase_sequence,layer_num])+".png"
        
        image_phase=cv2.imread(os.path.join(imagePhase_dir,phase_path)) 
        if patient_dir in second_devices:
            image_phase=255-image_phase  #对于第二批次的数据，需要进行反相处理，出血点在相位图中会变成白色
        
        image_phase = cv2.cvtColor(image_phase, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
        
        #为了保持顺序，一开始就对数据按照名称从小到大排序，可以通过index进行选择，而不是文件名，文件名的递增不是很好确定
        image_swis=[]
        # image_swis.append(image_swi)
        
        image_phases=[]
        # image_phases.append(image_phase)
        
        #如果该层图前面还有8层（包括当前层），后面还有8层，那么就可以进行处理，
        if index-7>=0 and index+8<=len(imgs_path)-1:#index就是该图像在所有图像中的按层数排序的位置
            for i in range(depth):  #验证没有问题
                Simage_depth_path=imgs_path[index-7+i]  
                image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                image_swis.append(image_Sdepth_data)
                
                deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                
                if patient_dir in second_devices: #将第一层相位图取反，接下来拼接层也进行取反
                    image_Pdepth_data=255-image_Pdepth_data
                
                image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                image_phases.append(image_Pdepth_data)
        
        else:  
            # 可能前面层不足8层（包括当前层）,则，后面层多取，直接连续取16层即可
            if index-7<0:
                for i in range(depth): 
                    Simage_depth_path=imgs_path[i]  # 直接从前向后遍历16层
                    image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                    image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                    image_swis.append(image_Sdepth_data)
                    
                    deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                    img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                    image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                    
                    if patient_dir in second_devices: #将第一层相位图取反，接下来拼接层也进行取反
                        image_Pdepth_data=255-image_Pdepth_data
                    
                    image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                    image_phases.append(image_Pdepth_data)
                
            elif index+8>len(imgs_path)-1: #后面曾不足8层，前面层多取，直接从后向前定位16层
                start=len(imgs_path)-1-15
                for i in range(start,len(imgs_path)): # 表示从start到len(imgs_path)-1
                    Simage_depth_path=imgs_path[i]  # 直接从前向后遍历16层
                    image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
                    image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                    image_swis.append(image_Sdepth_data)
                    
                    deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
                    img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
                    image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                    
                    if patient_dir in second_devices: #将第一层相位图取反，接下来拼接层也进行取反
                        image_Pdepth_data=255-image_Pdepth_data
                    
                    image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
                    image_phases.append(image_Pdepth_data)
                    
                
                
#region
#             #前面层顺序处理
#             for i in range(depth): 
#                 #最后16层没法处理   
#                 Simage_depth_path=imgs_path[index+i+1] #刚好边界达到图像的数目值
#                 image_Sdepth_data=cv2.imread(os.path.join(imageSwi_dir,Simage_depth_path)) 
#                 image_Sdepth_data = cv2.cvtColor(image_Sdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
#                 image_swis.append(image_Sdepth_data)
                
#                 deep_layer_num=Simage_depth_path.split("-")[-1].split(".")[0]
#                 img_phase_path="-".join(["img",phase_sequence,deep_layer_num])+".png"
#                 image_Pdepth_data=cv2.imread(os.path.join(imagePhase_dir,img_phase_path)) 
                
#                 if patient_dir in second_devices: #将第一层相位图取反，接下来拼接层也进行取反
#                     image_Pdepth_data=255-image_Pdepth_data
                
#                 image_Pdepth_data = cv2.cvtColor(image_Pdepth_data, cv2.COLOR_BGR2GRAY) #转换为单通道的灰度图
#                 image_phases.append(image_Pdepth_data)
                
#                 # image_swi=np.stack((image_swi,image_depth_data),axis=2)  #现在得到的是完整的图像，没有经过裁剪
#endregion
        image_swi=np.stack(image_swis,axis=2)  #现在得到的是完整的图像，没有经过裁剪   
        image_phase=np.stack(image_phases,axis=2)  #现在得到的是完整的图像，没有经过裁剪  
        if (image_swi.shape[2]!=16):
            print("error depth :",img_path,imageSwi_dir,image_swi.shape)
        if (image_phase.shape!=image_swi.shape):
            print("error shape:",img_path,imagePhase_dir,image_phase.shape)
        
        #根据层图像获取anno标注框信息，在叠加起来的图像上进行整体层面的裁剪
        with open(os.path.join(txt_dir,txt_file_name), 'r') as file:  #对于该层图像
            lines = file.readlines()  #每一行是一个标注框信息
            for j,line in enumerate(lines):  #针对每一个标注框生成一个3D数据信息
                numbers = line.strip().split()
                # 直接从txt文件中得到标注大小，会缺失掉图像原本尺寸大小
                id, x, y, w, h = numbers  #中心，宽度坐标表示 ；但是不同的机器扫描出来的图像尺寸大小不一样
                #没必要像下面这么写，上面cv读取图像的时候已经可以拿到图像的尺寸大小了
                width=512 if patient_dir not in second_devices else 216  #width为图像的宽度
                height=512 if patient_dir not in second_devices else 256   #height为图像的高度
                #从txt中拿到的一行数据，x_center(与width相关)，y_center(与height相关)，w(与width相关)，h(与height相关)
                # x, y, w, h = round(width*float(x)), round(height*float(y)), 20, 20
                x, y, w, h = round(width*float(x)), round(height*float(y)), round(float(w)*width), round(float(h)*height)
                x1=x-w//2 #x表示的列，y表示的行，从txt中的中心转换为
                y1=y-h//2
                x2=x+w//2
                y2=y+h//2

                #region
                anno_show_image=image_swi[:,:,0].copy()
                cv2.rectangle(anno_show_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框，宽度为2
                cv2.imwrite("temp/output_image_with_box3.png", anno_show_image)
                pp=1+1
                #不要在裁剪这部分设置固定的尺寸，不然选择最佳尺寸需要在源头调整比较大，可以在第二阶段读取时再调整
                #这里裁剪没有问题吧，对于图像裁剪是从第i行到第j行，第m列到第n列
                #endregion
                
                #在切片图像的时候是先行再列，下面的切片没有问题，y表示行，x表示列
                crop_swi = image_swi[y1 : y2, x1 : x2] #如果是tensor数据则超出边界不会报错,numpy也不会报错
                crop_phase=image_phase[y1 : y2, x1 : x2]
                
                first_layer_image = Image.fromarray(crop_swi[:,:,0].astype(np.uint8))
                # first_layer_image = Image.fromarray(anno_show_image[y1 : y2, x1 : x2].astype(np.uint8))
                # 保存图像
                # first_layer_image.save('temp/first_layer.png')
                
                #没有数据归一化操作,在CNN3D中进行，transform操作进行归一化
                crop_Data=np.hstack((crop_swi,crop_phase))
                
                if 0 not in crop_Data.shape:
                # if crop_Data.shape==(20,40,16):  #直接如果不是这个大小那么就不要，因为有的标注面积大小为0
                    # print("error :",img_path,imageSwi_dir,crop_Data.shape)
                    scaling = np.eye(4)
                    nii_img = nib.Nifti1Image(crop_Data, scaling)  #crop_data的数据值不会受到scaling的影响
                    #根据id来进行训练和测试的不同的保存
                    if patient_dir in second_devices:
                        resolution="low"
                    else:
                        resolution="high"
                    if id==str(0): #只有0号才是出血点，其余的都是干扰点，划分正样例和负样例
                        # j 是txt文件中第几行出现的标注框
                        nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/train_model/{resolution}/all/CMB/{patients}-{layer_num}-{j}.nii')    
                    else:
                        nib.save(nii_img,f'/mnt/hdd1/zhulu/hospital/second_stage/train_model/{resolution}/all/Non-CMB/{patients}-{layer_num}-{j}.nii')
            
for i, patient_dir in enumerate(tqdm(patients_dir, desc="Processing Patients", unit="patient")):

    niiData_build(patient_dir,"train")

# print(set(flag))
    # if i<25:
    #     niiData_build(patient_dir,"train")  #划分训练集和测试集
    # else:
    #     niiData_build(patient_dir,"test")
