import pandas as pd
from collections import defaultdict
import json

# 计算两个矩形框的重叠部分，返回重叠比例
def calculate_overlap(box1, box2):
    # 解包每个框的左上角和右下角坐标
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # 计算重叠区域的左上角和右下角
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    
    # 如果没有重叠部分，则返回0
    if x_overlap == 0 or y_overlap == 0:
        return 0
    
    # 计算重叠区域的面积
    overlap_area = x_overlap * y_overlap
    
    # 计算每个框的面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # 计算重叠面积占每个框的比例
    overlap_ratio1 = overlap_area / area1
    overlap_ratio2 = overlap_area / area2
    
    # 返回最小的重叠比例作为判断条件
    return min(overlap_ratio1, overlap_ratio2)

# 读取数据文件
file_path = 'secondStage/phase_pred/FromSwi_all_samples_predict.txt'
# file_path = 'secondStage/test_all_samples_predict.txt'
data = []

# 读取文件内容并处理
with open(file_path, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split('|')
        
        if len(parts) >= 4:
            patient_info = parts[0].strip()  # 病人信息
            actual_label = int(parts[1].strip())  # 实际标签
            predicted_label = int(parts[2].strip())  # 预测标签
            box = eval(parts[3].strip())  # 预测框
            logit=parts[4].strip()
            # 只保留预测为1的情况
            if predicted_label == 1:
                data.append((patient_info, actual_label, predicted_label, box,logit))

# 存储病人数据
patients = defaultdict(list)
final_result = {}
# 按照病人分组
for patient_info, actual_label, predicted_label, box,logit in data:
    # patient_id = patient_info.split("/")[1].split("-")[1]  # 获取病人ID
    patient_id = patient_info.split("/")[-1].split("-")[0]  # 获取病人ID
    patients[patient_id].append((patient_info, actual_label, predicted_label, box,logit))

def sort_key(item):
    filename = item[0]  # 获取元组的第一个元素（文件名）
    # 提取 '00092' 部分并转换为整数
    num_part = filename.split('-')[-2]  # 分割后取倒数第二部分
    return int(num_part)

# 给每个病人的出血点编号
for patient_id, results in patients.items():
    # 对病人的图像按图像编号排序
    results = sorted(results, key=sort_key)
    # results.sort(key=lambda x: int(x[0].split("-")[2]))  # 排序按图像编号
    
    
    bleeding_point_count = 1  # 记录当前病人的出血点编号
    merged_boxes = []  # 用来存储合并后的框

    for i in range(len(results)):
        current_image = results[i]
        current_box = current_image[3]
        current_image_num = int(current_image[0].split("-")[-2])  # 获取图像编号
        
        # 检查是否有与之前的框重叠超过50%，并且是连续的图像
        is_merged = False
        for j in range(len(merged_boxes)):  #会与每一个merged box进行检测
            # last_merged_image_num = int(merged_boxes[j][1][0][0].split("-")[2])  # 获取最后一个合并框的图像编号
            last_merged_image_num = int(merged_boxes[j][1][-1][0].split("-")[-2])  # Non-CMB会多一个"-"，因此从后向前数。获取最后一个合并框的图像编号
            
            # 判断是否为连续图像（图像编号差1）
            if abs(current_image_num - last_merged_image_num) == 1 and calculate_overlap(current_box, merged_boxes[j][0]) > 0.5:
                # 如果是连续图像且重叠超过50%，则合并该框并标记为同一个出血点
                merged_boxes[j][1].append(current_image)
                is_merged = True
                # last_merged_image_num=current_image_num
                break
        
        # 如果没有重叠框且没有合并，就认为这是一个新的出血点
        if not is_merged:
            merged_boxes.append([current_box, [current_image]])  # 保存新框及其对应图像
            bleeding_point_count += 1

    # # 打印出每个病人的出血点信息
    # print(f"Patient ID: {patient_id}")
    # for idx, (box, images) in enumerate(merged_boxes, 1):
    #     print(f"  Bleeding point {idx}:")
    #     for img in images:
    #         print(f"    {img[0]} - Predicted Box: {img[3]}")
    
    
    # 构建JSON格式的输出
    patient_result = []
    for idx, (box, images) in enumerate(merged_boxes, 1):
        # 每个出血点是一个字典，包含出血点编号和出血点相关的图像
        max_logits=max([float(img[-1]) for img in images])
        bleeding_point = {f'slices {idx}': [img[0] for img in images],"box":str(box),"logits":max_logits}
        patient_result.append(bleeding_point)
    
    # 将病人的出血点信息添加到最终结果中
    final_result[patient_id] = patient_result


output_file_path = 'output_blood_points_swi2phase.json'
# 将最终结果转换为JSON格式并输出
# 打开文件并写入JSON数据
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(final_result, f, ensure_ascii=False, indent=4)
