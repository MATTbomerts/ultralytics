import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import ast  
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR,CosineAnnealingLR
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import sys
import os
import random
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


from my_tool.utils import read_data,read_add_data




# print("Current file path:", os.path.abspath(__file__))

# # 打印当前工作目录
# print("Current working directory:", os.getcwd())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)  # 你可以选择任何数字作为随机种子，之前的测试准确率77%的准确率是随机种子，现在复现不出来了


def plot_loss_curve(train_loss,test_loss,save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    # 绘制测试损失曲线
    plt.plot(test_loss, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # 保存图像到文件
    plt.close()  # 关闭图像，释放内存

# 数据预处理
# 只对PIL或ndarray有效
transform = transforms.Compose([
    transforms.ToTensor(),  #操作会进行数据归一化
    transforms.Lambda(lambda x: x.permute(1, 2, 0))  #但是好像结合3D-CNN这个维度不需要转换
])


# 定义数据集类
class NiiDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)
    
    def add_data(self, new_file_paths, new_labels):
        """可以在初始化后增加数据"""
        self.file_paths.extend(new_file_paths)  # 添加新的文件路径
        self.labels.extend(new_labels)  # 添加新的标签
    

    def __getitem__(self, idx):
        # 返回拿到样本的图像数值数据和标签，但为了能拿到bad case 需要进一步得到数据的名称
        file_name="/".join(self.file_paths[idx].split("/")[-2:])
        nii_data = nib.load(self.file_paths[idx])
        img = nii_data.get_fdata()
        descrip = nii_data.header['descrip'].item().decode('utf-8')
        bbox_str = descrip.split('Bounding boxes: ')[1]  # 提取列表部分
        bbox_list = ast.literal_eval(bbox_str) 
        #得到的是numpy.memmap数据类型，后面的transform不会生效
        
        #原本的方式标注框大小不一致，因此需要进行缩放转换，现在不需要
        #region
        # img = img.astype(np.float32)  #原本的大小是20，40，16 #应该是不管什么样的大小都能进行转换
        # target_shape = (16, 32, 16) 
        # first_layer_image = Image.fromarray(img[:,:,7].astype(np.uint8))
        # # 保存图像
        # first_layer_image.save('temp/first_layer.png')
        # zoom_factors = np.array(target_shape) / np.array(img.shape)
        # img = zoom(img, zoom_factors)   #数据缩放的方式是插值操作，不是简单的裁剪
        # zoom_layer_image = Image.fromarray(img[:,:,7].astype(np.uint8))
        # # 保存图像
        # zoom_layer_image.save('temp/zoom_layer.png')
        #endregion
        
        img=img.astype(np.uint8)  #转换为float32类型
        swi_img=img[:,:16,:]  #前一半,确认是swi
        

        if self.transform:  #transforms.toTensor()操作会将形状为(H,W,depth)的数组转换为(depth,H,W)，其实是对的，因为3D-CNN输入第一个维度就是depth
            img = self.transform(swi_img)  # 已经经过归一化[0-1]之间浮点数
        label = self.labels[idx]  
        
        return img, label,file_name,torch.tensor(bbox_list)


class NiiDataset2(Dataset):
    def __init__(self, data, transform=None):
        self.data=data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img,label,file_name,pred_box = self.data[idx]
       
        img=img.astype(np.uint8)  #转换为uint8类型，再通过toTensor转换为【0-1】之间，归一化
        swi_img=img[:,:16,:]  #前一半
        if self.transform:  #transforms.toTensor()操作会将形状为(H,W,depth)的数组转换为(depth,H,W)，其实是对的，因为3D-CNN输入第一个维度就是depth
            swi_img = self.transform(swi_img)
        
        return swi_img, label,file_name,pred_box,img  #还要返回img是因为还要传给phase网络,img是没有经过归一化等处理的原始数据

#region
# 数据加载函数
def read_data1(class_names, class_labels,resolution):
    fold1_test, fold2_train, fold3_valid = [], [], []
    ts_lbl, tr_lbl, val_lbl = [], [], []

    for pos, sel in enumerate(class_names): # 将CMB和Non-CMB一起读出来，但是先读一种，再另一种，在dataloader中进行打乱
        
        images_test = sorted(glob.glob(f"/mnt/hdd1/zhulu/hospital/second_stage/train_model/{resolution}/test/{sel}/*.nii"))
        images_train = sorted(glob.glob(f"/mnt/hdd1/zhulu/hospital/second_stage/train_model/{resolution}/train/{sel}/*.nii"))
        images_valid = sorted(glob.glob(f"/mnt/hdd1/zhulu/hospital/second_stage/train_model/{resolution}/test/{sel}/*.nii"))

        for volume in images_test:
            fold1_test.append(volume)
            ts_lbl.append(class_labels[pos])

        for volume in images_train:
            fold2_train.append(volume)
            tr_lbl.append(class_labels[pos])

        for volume in images_valid:
            fold3_valid.append(volume)
            val_lbl.append(class_labels[pos])

    return fold2_train, fold1_test, fold3_valid, tr_lbl, ts_lbl, val_lbl
#endregion

# 定义3D-CNN模型
class CNN3D(nn.Module):
    def __init__(self, classes):
        super(CNN3D, self).__init__()
        #nn.Conv3d输入shape: batch,channel(3 RGB),depth,height,width
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1) #3d卷积核也没有问题，1指的是每层图像的通道数
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(128)
        self.fc1 = nn.Linear(128*2*2*2, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 128*2*2*2)  #如果输入拼接层数变少，这个维度也需要变化，以及上面全连接层的维度
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ComplexCNN3D(nn.Module):
    def __init__(self, classes):
        super(ComplexCNN3D, self).__init__()
        
        # 增加卷积层和使用残差连接
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        
        # 引入残差连接
        self.residual_block = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.BatchNorm3d(128)
        )
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 更复杂的全连接层
        self.fc1 = nn.Linear(256*2*2*2, 512)  # 增加更大的隐藏层
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, classes)
        
        # Dropout 和正则化
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 前向传播增加残差连接
        x1 = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x2 = torch.relu(self.bn2(self.conv2(x1)))
        x3 = self.pool(torch.relu(self.bn3(self.conv3(x2))))
        
        # 使用残差连接
        residual = self.residual_block(x3)
        x4 = torch.relu(self.bn4(self.conv4(x3 + residual)))  # 跳跃连接
        
        x5 = self.pool(torch.relu(self.bn5(self.conv5(x4))))
        
        # 扁平化
        x = x5.view(-1, 256*2*2*2)
        
        # 更复杂的全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # 输出层
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha    # 正样本权重系数
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma    # 难易样本调节因子
        self.reduction = reduction  # 损失聚合方式（'mean'/'sum'/'none'）

    def forward(self, inputs, targets):
        
        # 计算二分类交叉熵,none方式
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 计算预测的概率 pt，每个样本在真实类别上的预测概率
        softmax_preds = F.softmax(inputs, dim=-1)
        pt = torch.gather(softmax_preds, 1, targets.unsqueeze(1))
        # 使用类别特定的 alpha 权重
        device = inputs.device  # 获取 inputs 所在的设备
        self.alpha = self.alpha.to(device)
        alpha_t = self.alpha[targets]  # 按目标类别的索引获取权重
        # Focal Loss 核心公式
        focal_loss = alpha_t.unsqueeze(1) * torch.pow((1 - pt), self.gamma) * bce_loss.unsqueeze(1)
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return torch.mean(focal_loss) 
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        

# 训练和测试函数
def train_model(model, train_loader,test_loader, criterion, optimizer,scheduler, num_epochs,resolution):
    training_epoch_loss = []
    test_epoch_loss=[]
    best_loss=float("inf")
    no_improvement_count = 0
    min_delta=1e-4  #要大于一定的值时才算有进步
    # min_delta=0
    patience=10
    
    for epoch in tqdm(range(num_epochs),desc="training"):  #迭代轮数
        model.train()  #如果把下面的model.eval开启的话，需要将model.train放到这个位置，而不是这个for循环的外面
        running_loss = 0.0
        total=0
        correct=0
        total_train_predict_CMB_nums=0
        total_train_true_positive=0
        # for index,inputs, labels in enumerate(train_loader,desc="Training"):
        for inputs, labels , _,box in train_loader:  #一次所有的训练样本
            inputs, labels = inputs.cuda(), labels.cuda() #[batch,width,height(拼接)，depth]
            # print(inputs.shape)
            optimizer.zero_grad()
            inputs=inputs.reshape(inputs.shape[0], 1, 16, 16, 16) #输入维度没有问题，其中的1表示每层图像是一个灰度图
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predict_CMB_nums=sum(predicted==1)  #预测出来为CMB的数量
            total_train_predict_CMB_nums+=predict_CMB_nums
            # 真正为CMB的数量
            true_positive = ((predicted == 1) & (labels == 1)).sum().item()
            total_train_true_positive+=true_positive
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        
        training_loss=running_loss/len(train_loader)  #计算的是每个epoch之后的总体损失
        training_epoch_loss.append(training_loss)
        print(f"Epoch {epoch+1}, training Loss: {training_loss}; all class train Accuracy: {100 * correct / total:.2f}%;PPV: {100*total_train_true_positive/total_train_predict_CMB_nums:.2f}%")
        
        scheduler.step()
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Learning Rate: {current_lr}")

        
        model.eval()
        # 如果不加model.eval()，还会进行dropout影响性能，感觉有点过拟合了，训练集的损失在下降，但是测试集性能没有提升
        correct = 0
        total = 0
        running_loss_test=0.0
        
        total_predict_CMB_nums=0
        total_true_positive=0
        
        
        #把所有的训练数据跑完之后再进行的测试，在每个epoch结束之后再进行测评
        with torch.no_grad():
            for inputs, labels ,file_name,box in test_loader:  #一次所有的测试样本
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs=inputs.reshape(inputs.shape[0], 1, 16, 16, 16)
                outputs = model(inputs)
                loss_test = criterion(outputs, labels)
                
                #下面代码可以共用
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)  #这一批所有的数据
                correct += (predicted == labels).sum().item()  #这一batch中预测正确的个数（所有类别）
                # score = accuracy_score(predicted, labels)
                running_loss_test+=loss_test.item()
                
                predict_CMB_nums=sum(predicted==1)  #预测出来为CMB的数量
                total_predict_CMB_nums+=predict_CMB_nums
                #真正为CMB的数量，感觉有点问题，可能会漏掉标签为1，但实际不是1的数据
                true_positive = ((predicted == 1) & (labels == 1)).sum().item()  
                total_true_positive+=true_positive
                
        test_loss=running_loss_test/len(test_loader)
        test_epoch_loss.append(test_loss)
        print(f" test Epoch {epoch+1},test Loss: {test_loss};All Class Test Accuracy: {100 * correct / total:.2f}%; PPV:{100*total_true_positive/total_predict_CMB_nums:.2f}%\n")
        
        # Check if the test loss improved  并且要保证还是总体是上升的趋势，不是下降
        #如果最好的损失减去当前的损失大于最小的变化值(正值) 【当前的损失要小，因此将best_loss换为小的】
        #如果best_loss小但是当前test_loss大的话，就有一轮没有提升了
        if best_loss - test_loss > min_delta:  
            best_loss = test_loss
            torch.save(model.state_dict(), f"parameters/res_model/swi/CMB_3DCNN_lr0001_adamw_focal01_2_gamma2_multi_best.pth")
            no_improvement_count = 0  # Reset the counter if there's an improvement
        else:
            no_improvement_count += 1
        
        

        # # Early stopping condition  早停法设置
        # if no_improvement_count >= patience:
        #     print(f"Early stopping triggered after {epoch+1} epochs. No improvement in test loss for {patience} epochs. Best loss is {best_loss}")
        #     break
    
    torch.save(model.state_dict(), f"parameters/res_model/swi/CMB_3DCNN_lr0001_adamw_focal01_2_gamma2_multi.pth")    
    return training_epoch_loss,test_epoch_loss
        

def val_model(model, valid_loader):
    """ 这个方式中直接用numpy或者torch数据，不再经过nii数据的转换 """
    model.eval()   
    correct = 0
    total = 0
    
    total_val_predict_CMB_nums=0
    total_val_true_positive=0
    
    all_wrong_predict_files=[]
    all_files=[]
    all_labels=[]
    all_pred_labels=[]
    all_pred_boxes=[]
    with torch.no_grad():
        for inputs, labels ,file_name, pred_box in valid_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs=inputs.reshape(inputs.shape[0], 1, 16, 16, 16)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predict_CMB_nums=sum(predicted==1)  #预测出来为CMB的数量
            total_val_predict_CMB_nums+=predict_CMB_nums
            # 真正为CMB的数量
            true_positive = ((predicted == 1) & (labels == 1)).sum().item()
            total_val_true_positive+=true_positive
            
            wrong_index = (predicted != labels).nonzero()
            wrong_files=[file_name[i] for i in wrong_index]
            all_wrong_predict_files.extend(wrong_files)
            
            all_labels.extend(labels.tolist())
            all_pred_labels.extend(predicted.tolist())
            all_files.extend(file_name)  #file_name是一个list
            all_pred_boxes.extend(pred_box.tolist())
            
    print(f"test --- ALL Class Accuracy: {100 * correct / total:.2f}%; PPV:{100*total_val_true_positive/total_val_predict_CMB_nums:.2f}%")
    
    
    patient_dict={}
    for i in range(len(all_files)):
        patient_name="-".join(all_files[i].split("/")[-1].split("-")[:2])
        if patient_name not in patient_dict:
            patient_dict[patient_name]=[]
        patient_dict[patient_name].append((all_files[i],all_labels[i],all_pred_labels[i],all_pred_boxes[i]))
    
    
    patient_names=patient_dict.keys()
    for patient in patient_names:
        # """ 打印所有错误预测的样本 """
        # # print("wrong predict samples: ",all_wrong_predict_files)
        # with open("secondStage/wrong_predict_samples_{patient}.txt","w") as f:
        #     for item in all_wrong_predict_files:
        #         f.write(item)
        #         f.write("\n")
        
        """ 打印所有预测结果，真实标签 | 预测结果 """
        with open(f"secondStage/pred1/all_samples_predict_{patient}.txt","w") as f:
            for i in range(len(patient_dict[patient])):
                sample_name=patient_dict[patient][i][0]
                label=patient_dict[patient][i][1]
                pred_label=patient_dict[patient][i][2]
                pred_box=patient_dict[patient][i][3]
                f.write(f"{sample_name:<30}|{label:<5}|{pred_label:<5}|{pred_box}\n")
         

def test_model(model, test_loader):
    # 在test_model中加入是否进行可视化的代码
    # 将分类错误的样本保存下来，然后进行可视化
    model.eval()
    correct = 0
    total = 0
    all_wrong_predict_files=[]
    all_files=[]
    all_labels=[]
    all_pred_labels=[]
    all_boxes=[]
    
    all_logits=[]
    with torch.no_grad():
        for inputs, labels ,file_name,box in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  #初始得到的是 bsz,16,32,16
            inputs=inputs.reshape(inputs.shape[0], 1, 16, 16, 16)
            outputs = model(inputs)
            pred_logits,softmax_preds =torch.max(F.softmax(outputs, dim=-1),1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            wrong_index = (predicted != labels).nonzero()
            wrong_files=[file_name[i] for i in wrong_index]
            all_wrong_predict_files.extend(wrong_files)
            all_boxes.extend(box.tolist())
            all_labels.extend(labels.tolist())
            all_pred_labels.extend(predicted.tolist())
            all_files.extend(file_name)  #file_name是一个list
            pred_logits=pred_logits.tolist()
            pred_logits = [round(x, 2) for x in pred_logits]
            all_logits.extend(pred_logits)
            
    print(f"test --- Accuracy: {100 * correct / total:.2f}%")  #计算出的也是所有类别的准确率
    
    #region
    """ 打印所有错误预测的样本 """
    # print("wrong predict samples: ",all_wrong_predict_files)
    with open("secondStage/test_wrong_predict_samples.txt","w") as f:
        for item in all_wrong_predict_files:
            f.write(item)
            f.write("\n")
    
    """ 打印所有预测结果，真实标签 | 预测结果 """
    with open("secondStage/test_all_samples_predict.txt","w") as f:
        for i in range(len(all_files)):
            sample_name=all_files[i]
            label=all_labels[i]
            pred_label=all_pred_labels[i]
            box=all_boxes[i]
            logits=all_logits[i]
            f.write(f"{sample_name:<30}|{label:<5}|{pred_label:<5}|{str(box):<25}|{logits:<10}\n")
            # f.write(f"{sample_name:<30}|{label:<5}|{pred_label:<5}|{str(box):<25}\n")
    #endregion  
    
def predict_model(model, test_loader):
    # # 在test_model中加入是否进行可视化的代码
    # # 将分类错误的样本保存下来，然后进行可视化
    # model.eval()
   
    # pha_data=[]

    # with torch.no_grad():
    #     for inputs, labels ,file_name,box,raw in test_loader:
    #         inputs, labels = inputs.cuda(), labels.cuda()  #初始得到的是 bsz,16,32,16
    #         inputs=inputs.reshape(inputs.shape[0], 1, 16, 16, 16)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         for index, predict in enumerate(predicted):
    #             if predict==1:
    #                 pha_data.append([raw[index].cpu().numpy(),labels[index].cpu().numpy(),file_name[index],box[index].cpu().numpy()])
    # return pha_data
    # 在test_model中加入是否进行可视化的代码
    # 将分类错误的样本保存下来，然后进行可视化
    model.eval()
    correct = 0
    total = 0
    all_wrong_predict_files=[]
    all_files=[]
    all_labels=[]
    all_pred_labels=[]
    all_boxes=[]
    
    all_logits=[]
    pha_data=[]
    with torch.no_grad():
        for inputs, labels ,file_name,box,raw in test_loader:
            # inputs, labels = inputs.cuda(), labels.cuda()  #初始得到的是 bsz,16,32,16
            inputs, labels = inputs, labels  #初始得到的是 bsz,16,32,16
            inputs=inputs.reshape(inputs.shape[0], 1, 16, 16, 16)
            outputs = model(inputs)
            pred_logits,softmax_preds =torch.max(F.softmax(outputs, dim=-1),1)
            _, predicted = torch.max(outputs, 1)
            
            for index, predict in enumerate(predicted):
                if predict==1:
                    pha_data.append([raw[index].cpu().numpy(),labels[index].cpu().numpy(),file_name[index],box[index].cpu().numpy()])
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            wrong_index = (predicted != labels).nonzero()
            wrong_files=[file_name[i] for i in wrong_index]
            all_wrong_predict_files.extend(wrong_files)
            all_boxes.extend(box.tolist())
            all_labels.extend(labels.tolist())
            all_pred_labels.extend(predicted.tolist())
            all_files.extend(file_name)  #file_name是一个list
            pred_logits=pred_logits.tolist()
            pred_logits = [round(x, 2) for x in pred_logits]
            all_logits.extend(pred_logits)
            
    
    """ 打印所有预测结果，真实标签 | 预测结果 """
    with open("txt/test_all_samples_predict.txt","w") as f:
        for i in range(len(all_files)):
            sample_name=all_files[i]
            label=all_labels[i]
            pred_label=all_pred_labels[i]
            box=all_boxes[i]
            logits=all_logits[i]
            f.write(f"{sample_name:<30}|{label:<5}|{pred_label:<5}|{str(box):<25}|{logits:<10}\n")
            # f.write(f"{sample_name:<30}|{label:<5}|{pred_label:<5}|{str(box):<25}\n")
    #endregion  
    return pha_data

def CMB_3DCNN_main(resolution):
    class_names = ['Non_CMB', 'CMB']
    class_labels = [0, 1]  #Non-CMB:0, CMB:1
    #这部分数据没有读错
    datas_files,labels = read_data(class_names, class_labels,resolution)
    
    train_files, test_files, train_labels, test_labels = train_test_split( #根据标签比例将数据划分为训练集和测试集
    datas_files,            # 输入数据
    labels,           # 标签
    test_size=0.2,    # 测试集所占比例，可根据需要调整
    stratify=labels,  # 确保每类标签在划分后保持相同的比例
    random_state=42   # 设置随机种子，保证结果可复现
)

    # 此处的train_files是一个引用传递，在后面add files之后，该train files也变了
    train_dataset = NiiDataset(train_files, train_labels, transform) 
    test_dataset = NiiDataset(test_files, test_labels, transform)
    # valid_dataset = NiiDataset(valid_files, val_lbl, transform)
    

    # 假设我们要添加一些新的增强之后的数据到现有的Dataset，此种方式无法再划分到训练和测试集中
    add_dir="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add3"
    add_datas_files,add_labels = read_add_data(class_names, class_labels,add_dir,resolution)
    add_train_files, add_test_files, add_train_labels, add_test_labels = train_test_split(
        add_datas_files,            # 输入数据
        add_labels,           # 标签
        test_size=0.2,    # 测试集所占比例，可根据需要调整
        stratify=add_labels,  # 确保每类标签在划分后保持相同的比例
        random_state=42 
    )
    
    
    # # 添加多点数据
    # add_dir="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/multi_blood"
    # add_datas_files1,add_labels1 = read_add_data(class_names, class_labels,add_dir,resolution)
    # add_train_files1, add_test_files1, add_train_labels1, add_test_labels1 = train_test_split(
    #     add_datas_files1,            # 输入数据
    #     add_labels1,           # 标签
    #     test_size=1e-10,    # 测试集弄比较小，尽量全部数据用于训练
    #     stratify=add_labels1,  # 确保每类标签在划分后保持相同的比例
    #     random_state=42 
    # )
    
    
    # new_dataset = NiiDataset(add_datas_files, add_labels, transform)
    train_dataset.add_data(add_train_files, add_train_labels)
    test_dataset.add_data(add_test_files, add_test_labels)
    
    # train_dataset.add_data(add_train_files1, add_train_labels1)
    # test_dataset.add_data(add_test_files1, add_test_labels1)
    
    
    # add_dir1="/mnt/hdd1/zhulu/hospital/second_stage/Yolo23d/high/add1"
    # add_datas_files1,add_labels1 = read_add_data(class_names, class_labels,add_dir1,resolution)
    # # new_dataset = NiiDataset(add_datas_files, add_labels, transform)
    # train_dataset.add_data(add_datas_files1, add_labels1)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 进行数据打乱
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # train_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)  # 进行数据打乱
    # test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

    # model = CNN3D(2).cuda()
    model = ComplexCNN3D(2).cuda()
    
    # 定义类别权重，对于CMB出血点应该给予更高的权重
    class_weights = torch.tensor([1.0, 60.0]).cuda()  # Non-CMB 的权重是 1.0，CMB 的权重是 5.0，与标签顺序一致
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss(alpha=0.05,gamma=2)
    criterion = FocalLoss(alpha=[0.1,2],gamma=2)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  #将优化器改为adamw
    # scheduler = ExponentialLR(optimizer, gamma=0.8)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)  #迭代器换成余弦

    if Train:
        train_loss,test_loss=train_model(model, train_loader,test_loader, criterion, optimizer, scheduler,num_epochs=50,resolution=resolution)
        # torch.save(model.state_dict(), "parameters/CMB_3DCNN.pth")
        plot_loss_curve(train_loss,test_loss,save_path='secondStage/figs/loss_curve_norm_lr_exp.png')
        
    if Test:
        # model.load_state_dict(torch.load(f"parameters/res_model/swi/CMB_3DCNN_lr0001_90_adamw_focal75_gamma2_none.pth"))
        model.load_state_dict(torch.load(f"parameters/res_model/swi/CMB_3DCNN_lr0001_90_adamw_focal90_gamma2_none_last.pth"))
        test_model(model, test_loader)
    
    if Val:
        model.load_state_dict(torch.load("parameters/CMB_3DCNN_new1.pth"))
        val_model(model, valid_loader)

if __name__ == "__main__":
    # Train = True
    Train = False
    
    Test = True
    # Test=False
    
    # Val=True
    Val=False
    resolution="high"
    CMB_3DCNN_main(resolution)
