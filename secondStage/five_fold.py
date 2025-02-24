import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR
import nibabel as nib
from sklearn.metrics import accuracy_score
from scipy.ndimage import zoom
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset

import random
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

    def __getitem__(self, idx):
        # 返回拿到样本的图像数值数据和标签，但为了能拿到bad case 需要进一步得到数据的名称
        file_name="/".join(self.file_paths[idx].split("/")[-2:])
        img = nib.load(self.file_paths[idx]).get_fdata()
        #得到的是numpy.memmap数据类型，后面的transform不会生效
        img = img.astype(np.float32)  #原本的大小是20，40，16 #应该是不管什么样的大小都能进行转换
        target_shape = (16, 32, 16)
        
        first_layer_image = Image.fromarray(img[:,:,7].astype(np.uint8))
        # 保存图像
        first_layer_image.save('temp/first_layer.png')

        zoom_factors = np.array(target_shape) / np.array(img.shape)
        img = zoom(img, zoom_factors)   #数据缩放的方式是插值操作，不是简单的裁剪
        
        zoom_layer_image = Image.fromarray(img[:,:,7].astype(np.uint8))
        # 保存图像
        zoom_layer_image.save('temp/zoom_layer.png')
        
        img=img.astype(np.uint8)  #转换为float32类型
        
        if self.transform:  #transforms.toTensor()操作会将形状为(H,W,depth)的数组转换为(depth,H,W)，其实是对的，因为3D-CNN输入第一个维度就是depth
            img = self.transform(img)  # 已经经过归一化[0-1]之间浮点数
        label = self.labels[idx]  
        
        return img, label,file_name


class NiiDataset2(Dataset):
    def __init__(self, data, transform=None):
        self.data=data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img,label,file_name,pred_box = self.data[idx]
        img=img.astype(np.float32)  #原本的大小是20，40，16
        target_shape = (16, 32, 16)  #如果是在这里进行3D形状的转换，那么在第一阶段的输出中就不用再管了
        
        zoom_factors = np.array(target_shape) / np.array(img.shape)
        img = zoom(img, zoom_factors)   #数据缩放的方式是插值操作，不是简单的裁剪
        img=img.astype(np.uint8)  #转换为uint8类型，再通过toTensor转换为【0-1】之间，归一化
        if self.transform:  #transforms.toTensor()操作会将形状为(H,W,depth)的数组转换为(depth,H,W)，其实是对的，因为3D-CNN输入第一个维度就是depth
            img = self.transform(img)
        
        return img, label,file_name,pred_box

def save_loss_curve(all_losses, filename="training_testing_loss_curve_weight.png"):
    num_folds = len(all_losses)
    plt.figure(figsize=(15, 10))  # 设置较大画布，以便包含多个子图
    
    for i, (train_losses, test_losses) in enumerate(all_losses):
        plt.subplot(2, 3, i + 1)  # 创建子图，2行3列布局
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Fold {i+1} Loss Curve')
        plt.legend()
    
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(filename)  # 保存图像文件
    plt.close()  # 关闭图表以释放内存


# 数据加载函数
def read_data(class_names, class_labels,resolution):
    #class_names:[CMB,Non-CMB]
    data_fold=[]
    label=[]
    # 从不同的文件夹读数据时，文件夹名就是标签label
    for pos, sel in enumerate(class_names): # 将CMB和CMB一起读出来，但是先读一种，再另一种，在dataloader中进行打乱
        
        datas=sorted(glob.glob(f"/mnt/hdd1/zhulu/hospital/second_stage/train_model/{resolution}/all/{sel}/*.nii"))
        for data in datas:
            data_fold.append(data)
            label.append(class_labels[pos])
    return data_fold,label

# 定义3D-CNN模型
class CNN3D(nn.Module):
    def __init__(self, classes):
        super(CNN3D, self).__init__()
        #nn.Conv3d输入shape: batch,channel(3 RGB),depth,height,width
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
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
        self.fc1 = nn.Linear(128*2*4*2, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 128*2*4*2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        
def CMB_3DCNN_main(resolution):
    class_names = ['Non-CMB', 'CMB']
    class_labels = [0, 1]  #Non-CMB:0, CMB:1
    #读出总的数据，只是文件路径
    datas,labels = read_data(class_names, class_labels,resolution)
    all_losses = []  # 用于保存所有折的损失曲线
    test_losses=[]

    # 使用 StratifiedKFold 进行 5 折分层交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #的确能将数据按照类别划分为两大类
    all_scores = []
    fold_score=[]  #保存每一折最后一轮迭代的准确率
    softmax = nn.Softmax(dim=1)
    for fold, (train_idx, test_idx) in enumerate(skf.split(datas, labels)): #split对Datas并不检查数据类型，可以是文本
        print(f'Fold {fold+1}')
        
        # 存储每一折的损失曲线
        fold_losses = []
        test_fold_loss=[]
        # 创建模型实例、损失函数和优化器  
        # 每一折都进行独立的训练
        model = CNN3D(2).cuda()
        class_weights = torch.tensor([1.0, 5.0]).cuda()
        #得到的是平均损失，因此最后除以len(loader)，应该也没有多大的问题
        criterion = nn.CrossEntropyLoss(weight=class_weights)  
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        scheduler = ExponentialLR(optimizer, gamma=0.90)
        # 创建数据加载器
        train_dataset = Subset(NiiDataset(datas, labels,transform), train_idx)
        test_dataset = Subset(NiiDataset(datas, labels,transform), test_idx)
        train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
        model.train()
        epochs=100
        for epoch in tqdm(range(epochs),desc="training"):
            running_loss = 0.0  #计算的是每个epoch的平均损失
            y_train_true, y_train_pred = [], []
            train_loader_num=0
            for X_batch, y_batch,file_name in train_loader:
                #只能保证整个epoch中两个类别的数据比例是一致的，但是不能保证每个batch中的比例是一致的
                X_batch=X_batch.cuda()
                y_batch=y_batch.cuda()
                optimizer.zero_grad()

                X_batch=X_batch.reshape(X_batch.shape[0], 1, 16, 32, 16)
                outputs = model(X_batch)
                probabilities = softmax(outputs)
                
                _, predicted_label = torch.max(outputs, 1)  #得到预测出来的标签
                
                y_train_true.extend(y_batch.tolist())  #将每个batch的结果拼接，成为整个训练集结果
                y_train_pred.extend(predicted_label.tolist())
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()  
                running_loss += loss.item() 
                
                train_loader_num+=1
            
            
            #看到底哪些位置预测存在较大的错误，再去选择看哪个test_loader
            difference_mask_train = torch.tensor(y_train_true) != torch.tensor(y_train_pred)
            different_indices_train = torch.nonzero(difference_mask_train).squeeze()
                 
            epoch_loss = running_loss / len(train_loader)
            fold_losses.append(epoch_loss)  # 添加每个 epoch 的平均损失
            scheduler.step()  #在一轮迭代结束后，再进行学习率调整
            
            
            # 模型评估 ，每个epoch之后的验证集损失
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                running_test_loss=0  #在进行一个epoch之后便清零
                loader_num=0  # 每次都比较以下对应位置的数据，看看损失为什么会变化
                for X_batch, y_batch,file_name in test_loader:
                    X_batch=X_batch.cuda()
                    y_batch=y_batch.cuda()
                    X_batch=X_batch.reshape(X_batch.shape[0], 1, 16, 32, 16)
                    outputs = model(X_batch)
                    
                    #用于观察结果
                    probabilities = softmax(outputs)
                    
                    test_loss = criterion(outputs, y_batch)
                    running_test_loss += test_loss.item() 
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(y_batch.tolist())  #将每个batch的结果拼接，成为整个测试集结果
                    y_pred.extend(predicted.tolist())
                    
                    loader_num+=1
                
                #看到底哪些位置预测存在较大的错误，再去选择看哪个test_loader
                difference_mask = torch.tensor(y_true) != torch.tensor(y_pred)
                different_indices = torch.nonzero(difference_mask).squeeze()
                
                epoch_test_loss = running_test_loss / len(test_loader)
                test_fold_loss.append(epoch_test_loss)  
                print(f'Epoch {epoch} Test Loss: {epoch_test_loss:.4f}')
            #计算的是整个测试集的预测准确率，而不是迭代中的单个batch
            score = accuracy_score(y_true, y_pred)
            print(f'Accuracy for fold {fold+1}: {score:.4f},loss {epoch_test_loss:.4f} in epoch {epoch}')  #每个epoch的准确率
            if epoch==epochs-1:  #保存最后一个epoch的准确率
                fold_score.append(score)  
            
        all_losses.append((fold_losses, test_fold_loss))
        
    avg_score = sum(fold_score)/len(fold_score)  #all_scores包含了每一折的平坔准确率
    print(f'5-Fold Cross-Validation Average Score: {avg_score:.4f}')
    print(f'5-Fold Cross-Validation  Score: {fold_score}')

    save_loss_curve(all_losses, filename="training_loss_curve_weight_5.png")
    

    

if __name__ == "__main__":
    # Train = True
    Train = False
    
    Test = True
    # Test=False
    
    # Val=True
    # Val=False
    resolution="high"
    CMB_3DCNN_main(resolution)
