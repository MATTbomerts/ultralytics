import json
import os
from tqdm import tqdm
import shutil

""" 根据annotations.json文件来进行patient下label中txt文件的生成，原本有txt文件不影响，会先删除再生成 """

micro_blood_num=0   #统计一下一共有多少个出血点
#在这里面将标注框改成20 pixel x 20 pixel大小
def convert_coco_to_yolo(coco_annotation_file, output_dir):
    # 加载COCO标注文件
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 获取类别信息
    categories = coco_data['categories']
    #在项目中的category['name']是数字编码，不是文本字符串
    category_map = {category['id']: category['name'] for category in categories}

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 遍历每个图像
    for image in tqdm(coco_data['images']): #images为一个列表，从image列表来创建txt文件，而不是annotation，但coco中不是每个图像都有annotation
        image_id = image['id']
        image_filename = image['file_name']
        image_width = image['width'] #图像宽度
        image_height = image['height'] #图像高度

        # YOLO 标注文件名
        yolo_filename = os.path.splitext(image_filename)[0] + '.txt' #为每个图像创建一个txt文件，图像的编号从几开始就是几
        yolo_filepath = os.path.join(output_dir, yolo_filename)
        
        
        yolo_annotations = []
        
        # 获取当前图像的所有标注
        #对json中所有的annotations进行遍历，找到针对这张图像的标注
        for annotation in coco_data['annotations']: 
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id']  #是整数，不是字符
                if category_id==1:
                    global micro_blood_num
                    micro_blood_num+=1
                category_name = category_map[category_id]
                
                # 获取标注的边界框
                bbox = annotation['bbox']
                # x是水平方向的坐标表示列数，y是垂直方向的坐标表示行数，因此后面x是与图像宽度进行一起处理
                x, y, w, h = bbox  #这里是左上角x,y坐标，宽度和高度，这里的xy怎么理解的？谁是行，谁是列？
                

                # 计算YOLO格式的中心点和宽高比例，中心点的计算还是要看原来标注的宽高
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                
                #手动修改目标框的大小：20 x 20，对于某些标注出血点不在框中心的点，可能调整之后框中没有目标点
                # w,h=20,20
                width = w / image_width 
                height = h / image_height
                
                yolo_annotation = f"{category_id - 1} {x_center} {y_center} {width} {height}"
                yolo_annotations.append(yolo_annotation)
                
        if yolo_annotations:
            with open(yolo_filepath, 'w') as yolo_file:
                for annotation in yolo_annotations:
                    yolo_file.write(annotation + '\n')
                

if __name__ == "__main__":  
    label_count=0 #表示包含label的文件有多少个
    parent_directory = '/mnt/hdd1/zhulu/blood_stage2/PNG-1.14'  # 替换为你的目录路径
    patient_dirs=os.listdir(parent_directory)
    subdirectory_name = 'swi/label'  #外面还有一个目录 "label"
    for patient in patient_dirs:
        patient_annoPath = os.path.join(parent_directory, patient,subdirectory_name,"COCO",'annotations.json')
        if os.path.exists(patient_annoPath):
            label_count+=1
            output_dir= os.path.join(parent_directory, patient, 'label')
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)  #删除原来的文件夹,再创建新的文件夹，确保数据不会新旧混叠
            # print("output_dir:", output_dir)
            convert_coco_to_yolo(patient_annoPath, output_dir)
            
        else:
            print(f"Annotation file not found for patient {patient}")
        
print("micro_blood_num:",micro_blood_num)
print("label_count:",label_count)   
