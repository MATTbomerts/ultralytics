import os
import ast
import cv2
import numpy as np

file_dir="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
pred_file="filtered.txt"
swi="Series-Ax SWAN new"

# Create a function to find companion points and update the matrix
def find_and_update(matrix, start_point, threshold=2):
    #region
    # height, width = matrix.shape
    # visited = np.zeros_like(matrix, dtype=bool)  # To track visited points
    # result_matrix = np.full_like(matrix, 255, dtype=np.uint8)  # Initialize with 255
    # queue = [start_point]  # Start from the given point
    # visited[start_point] = True

    # # Directions for 4-neighborhood
    # directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # while queue:
    #     x, y = queue.pop(0)
    #     result_matrix[x, y] = 0  # Set companion points to 0

    #     for dx, dy in directions:
    #         nx, ny = x + dx, y + dy
    #         if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
    #             if abs(int(matrix[nx, ny]) - int(matrix[x, y])) <= threshold:
    #                 visited[nx, ny] = True
    #                 queue.append((nx, ny))
    #endregion
    
    height, width = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)  # To track visited points
    result_matrix = np.full_like(matrix, 255, dtype=np.uint8)  # Initialize with 255
    queue = [start_point]  # Start from the given point
    visited[start_point] = True

    # Get the seed point value
    seed_value = matrix[start_point]

    # Directions for 4-neighborhood
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.pop(0)
        result_matrix[x, y] = 0  # Set companion points to 0

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                # Compare with the seed point value
                if abs(int(matrix[nx, ny]) - int(seed_value)) <= threshold:
                    visited[nx, ny] = True
                    queue.append((nx, ny))

    return result_matrix

def find_min_in_center(matrix, margin=5):
    height, width = matrix.shape
    # Restrict the search area to avoid the edges
    restricted_area = matrix[margin:height-margin, margin:width-margin]
    # Find the minimum value and position within the restricted area
    min_value = np.min(restricted_area)
    min_position = np.unravel_index(np.argmin(restricted_area), restricted_area.shape)
    # Adjust position to match the original matrix indices
    min_position = (min_position[0] + margin, min_position[1] + margin)
    return min_value, min_position


def calculate_circularity(binary_image):
    # Find contours
    binary_image = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to store the largest contour properties
    largest_contour = None
    largest_area = 0
    if len(contours) <= 3 and len(contours[0]) <= 3:
        return 1
    
    # Iterate through contours to find the largest one
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    
    # Calculate circularity if a contour is found
    circularity = None
    if largest_contour is not None:
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:  # Avoid division by zero
            circularity = (4 * np.pi * largest_area) / (perimeter ** 2)
    
    return circularity

read_dir="secondStage/filtered"
for pred_file in os.listdir(read_dir):
    if "Patient" not in pred_file:
        continue
    with open(os.path.join(read_dir,pred_file),"r") as f:
        #只处理预测为出血点的情况
        lines = f.readlines()
        shape_filtered_lines = []
        for index,line in enumerate(lines):
            data_path,real,pred,box=line.strip().split("|")  #每一行就是一个box，而不是一个层面的所有box
            real=real.strip()
            pred=pred.strip()
            if pred=="0" and real=="0":   #有可能会出现把出血当作非出血
                continue
            if pred=="0" and real=="1":
                shape_filtered_lines.append(line)
                continue
            data_type,patient_layer=data_path.split("/")[-2:]
            patient="-".join(patient_layer.split("-")[:2])
            layer=patient_layer.split("-")[-2]
            box=ast.literal_eval(box)
            #在进行验证的时候,并没有保存下来nii数据,因此还是通过原本图像来进行验证
            parts=data_path.split("/")[-1]
            patient_name,layer=parts.split("-")[1:3]
            img_name="img-00005-"+layer+".png"
            img_file_path=os.path.join(file_dir,"Patient-"+patient_name,swi,img_name)
            image=cv2.imread(img_file_path,cv2.COLOR_BGR2GRAY)
            image_draw=image.copy()
            cv2.rectangle(image_draw,(box[0][0],box[0][1]),(box[0][2],box[0][3]),(0,0,255),1)  #红色
            cv2.imwrite(f"temp/box.png",image_draw)
            x1, y1, x2, y2 = box[0]
            cropped_image = image[y1:y2, x1:x2]  # 使用numpy切片裁剪
            min_value, min_position = find_min_in_center(cropped_image)
            
            # min_value = np.min(cropped_image)
            # min_position = np.unravel_index(np.argmin(cropped_image), cropped_image.shape)

            matrix=find_and_update(cropped_image,min_position,10)
            cv2.imwrite(f"temp/binary.png",matrix)
            
            circularity=calculate_circularity(matrix)
            if circularity>=0.6:
                shape_filtered_lines.append(line)
            
            #region
            # sorted_array = np.sort(cropped_image.flatten())[::-1]  #从大到小进行排序
            # sorted_array=sorted_array.astype(np.int16)
            # diff=np.diff(sorted_array)
            # indices = np.where(diff <= -2)[0] + 1
            # if indices.size==0:
            #     continue
            
            # if indices.size==1:
            #     if indices>250:
            #         threshold_value=50
            # else:
            #     flag=0
            #     for idx in indices:
            #         if 100 < idx < 250 and sorted_array[idx]<100:
            #             threshold_value = sorted_array[idx]
            #             flag=1
            #             break
                    
            #     if flag==0:
            #         threshold_value = 50
            #     # if indices[1]>250:
            #     #     threshold_value = sorted_array[indices[0]]
            #     # else:
            #     #     threshold_value = sorted_array[indices[1]]  # 设置阈值
            
            # max_value = 255  # 最大值
            # # 应用阈值
            # _, binary_image = cv2.threshold(cropped_image, threshold_value, max_value, cv2.THRESH_BINARY)
            
            # cv2.imwrite(f"temp/box1.png",binary_image)
            # p=1+1
            # sp=1+2
            #endregion
    out_dir="secondStage/shape"
    with open(f"{out_dir}/{pred_file}_shape.txt","w") as out_f:
        for shape_filtered_line in shape_filtered_lines:
            out_f.write(shape_filtered_line)
            