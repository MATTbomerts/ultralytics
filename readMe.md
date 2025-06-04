## 环境安装

### 创建python环境，版本(3.9.19)
conda create -n myenv python=3.7   

conda activate myenv 

### 安装环境依赖
pip install -r requirements.txt

### 去头骨HD-BET安装
1：git clone https://github.com/MIC-DKFZ/HD-BET  

2：pip install -e .  

详细见 https://github.com/MIC-DKFZ/HD-BET


## 数据准备

|--data

|   |--patient1

|    |    |--swi

|    |    |--pha

|    |--patient2



## 执行指令

python main.py \

--dicom_dir data

--nii_dir  #保存中间数据nii的路径

--png_dir #保存中间数据png的路径



## 输出结果

|--results

|   |--patient1

|   |   |--results.json

## 注意：
1：需在根目录下创建txt文件夹，保存预测结果