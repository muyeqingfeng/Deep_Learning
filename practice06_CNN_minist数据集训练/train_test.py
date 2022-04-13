#  Name : Weishuo
#  Time : 2022/4/8 14:20
# E-mail: muyeqingfeng@126.com

""" 引入库函数 """
import torch
import matplotlib.pyplot as plt
from torch import nn, optim            # 优化模块，封装了求解模型的一些优化器如Adam,SGD
from torchvision import datasets       # pytorch 视觉库提供了加载数据集的接口
from torchvision import transforms     # pytorch 视觉库中提供了一些数据变换的接口
import torch.nn.functional as F          # 提供了一些常用的函数，如softmax
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torch.utils.data import DataLoader
from net import ConvNet

""" 设置超参数 """
# 预设网络超参数
batch_size = 50
# 让torch判断是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3

""" 加载数据集 """
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='..\data', train=True,
                               transform=data_transform, download=False)
test_dataset = datasets.MNIST(root='..\data', train=False,
                             transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

""" 训练前准备 """
# 初始化模型， 将网格操作移动到GPU或者CPU
ConvModel = ConvNet().to(device)
#　定义交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)
#　定义模型优化器
optimizer = optim.Adam(ConvModel.parameters(), lr=learning_rate)
# 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
# 如果学习率lr=0.05, 衰减周期step_size为30，耍贱乘法因子gamma=0.01
# Assuming optimizer uses lr=0.05 for all groups
# >>> # lr = 0.05     if epoch < 30
# >>> # lr = 0.005    if 30 <= epoch < 60
# >>> # lr = 0.0005   if 60 <= epoch < 60

""" 训练 """
def train(num_epochs, _model, _device, _train_loader, _optimizer, _lr_scheduler):
    # 设置模型为训练模式
    _model.train()
    # 设置学习率调度器开始准备更新
    _lr_scheduler.step()
    for epoch in range(num_epochs):
        # 从迭代器抽取图片和标签
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)
            # 此时样本是一批图片，在CNN的输入中，我们需要将其变为四维，
            # reshape第一个-1 代表自动计算批量图片的数目n
            # 最后reshape得到的结果就是n张图片，每一行图片都是单通道的28*28，得到的四维张量
            output = _model(samples.reshape(-1, 1, 28, 28))
            # 计算损失函数
            loss = criterion(output, labels)
            # 优化器内部参数梯度清零
            optimizer.zero_grad()
            # 损失值后向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch+1,
                                            num_epochs, i+1, loss.item()))

""" 测试 """
def test(_test_loader, _model, _device):
	# 设置模型进入评估模式
	_model.eval()
	loss = 0
	correct = 0

	with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源
		for data, target in _test_loader:
			data, target = data.to(_device), target.to(_device)
			output = ConvModel(data.reshape(-1, 1, 28, 28))
			loss += criterion(output, target).item()  # 添加损失值
			pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu

	loss /= len(_test_loader.dataset)

	print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
		loss, correct, len(_test_loader.dataset),
		100. * correct / len(_test_loader.dataset)))

""" 运行 """
for epoch in range(1, 4):
    train(epoch, ConvModel, device, train_loader, optimizer, exp_lr_scheduler)
    test(test_loader, ConvModel, device)
    test(train_loader, ConvModel, device)
