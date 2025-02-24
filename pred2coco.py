import json

# COCO JSON 数据结构初始化
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "0","color": [
                244,
                108,
                59
            ]},  # 固定映射：true_label 0 -> category_id 1
        {"id": 2, "name": "1","color": [
                99,
                102,
                129
            ]}   # 固定映射：true_label 1 -> category_id 2
    ]
}

# 固定类别映射
category_mapping = {
    "0": 1,  # true_label 为 "0"，对应 category_id 为 1
    "1": 2   # true_label 为 "1"，对应 category_id 为 2
}

# 读取 txt 文件
input_file = "/home/zhulu/workspace/ultralytics/secondStage/phase_pred/all_samples_predict_Patient-0000137786.txt"  # 替换为你的文件路径
annotation_id = 1
image_id = 1

with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        # 按 | 分割数据
        parts = line.strip().split('|')
        data_path, true_label, pred_label, bounding_box = parts
        true_label=true_label.strip()
        # 检查 true_label 是否在定义的 category_mapping 中
        if true_label not in category_mapping:
            raise ValueError(f"Unexpected true_label: {true_label}. Expected '0' or '1'.")
        
        # 解析 bounding_box
        bounding_box = eval(bounding_box)  # 将字符串转换为列表或元组
        bounding_box = [bounding_box]  # 如果 bounding_box 是一组坐标，将其封装为列表以便迭代
        
        img_name=data_path.split("/")[-1].split("-")[-2]
        # 添加图片信息到 COCO 的 images 字段
        coco_data["images"].append({
            "id": image_id,
            "file_name": "img-00005-"+img_name+".png",  # 替换为你的图片文件名
        })
        
        # 添加标注信息到 COCO 的 annotations 字段
        for bbox in bounding_box:
            # bbox 是 [x_min, y_min, x_max, y_max]，需要转换为 [x, y, width, height]
            x_min, y_min, x_max, y_max = bbox
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[true_label],  # 使用固定映射
                "bbox": [x, y, w, h],  # 转换后的 COCO 格式 bbox
                "area": w * h,  # 计算面积
                "iscrowd": 0   # 通常设置为 0
            })
            annotation_id += 1
        
        # 增加 image_id
        image_id += 1

# 输出 COCO 格式的 JSON 文件
output_file = "output.json"
with open(output_file, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO JSON 文件已生成: {output_file}")
