#  Name : Weishuo
#  Time : 2022/4/2 13:35
# E-mail: muyeqingfeng@126.com

"""
1.定义一个类，这个类是你的模型，你定义的这个类的父类是（torch.nn.Module）,
至少，最好必须要有两个函数，构造函数_init_ 用来初始化你的对象和forward用来做前馈运算。
Module这个类已经默认自动帮你算反向传播过程了。你这个模型用到的是类是nn.Linear,这个类做的运算是y=Ax+b. bias就是b
2.代价函数是：mse(torch.nn.MSELoss)
3.优化器选择（optimizer）,随机梯度下降算法SGD（torch.optim.SGD）
4.搭配上你之前的数据生成器来完成模型的训练，利用之前x,y
5.总结来说就是：第一步算前馈过程的预测值；第二步算损失；第三步算backward，在算backward之间梯度清零；第四步，更新。
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class linear_first(nn.Module):
	# 定义网络层信息
	def __init__(self, input_dim):
		super(linear_first, self).__init__()
		self.line1 = nn.Linear(input_dim, 1)
	# 构建前向传播计算过程
	def forward(self, x):
		y = self.line1(x)
		return y

class linear_second(nn.Module):
	# 定义网络层信息
	def __init__(self, input_dim):
		super(linear_second, self).__init__()
		self.line1 = nn.Linear(input_dim, 64)
		self.line2 = nn.Linear(64, 1)
	# 构建前向传播计算过程
	def forward(self, x):
		y = Fun.relu(self.line1(x))
		y = self.line2(y)
		return y

class linear_third(nn.Module):
	# 定义网络层信息
	def __init__(self, input_dim):
		super(linear_third, self).__init__()
		self.line1 = nn.Linear(input_dim, 64)
		self.line2 = nn.Linear(64, 128)
		self.line3 = nn.Linear(128, 1)
	# 构建前向传播计算过程
	def forward(self, x):
		y = Fun.relu(self.line1(x))
		y = Fun.relu(self.line2(y))
		y = self.line3(y)
		return y



""" 生成数据 """
class TensorDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return len(self.x)

np.random.seed(1)
w = 2.0
b = 6.0
nd = torch.Tensor([[i] for i in np.random.randint(-20, 20, 100)])
x_data = torch.Tensor([[i] for i in np.random.randint(1, 100, 100)])
y_data = w * x_data + b + nd
tensor_dataset = TensorDataset(x_data, y_data)

tensor_dataloader = DataLoader(tensor_dataset, # 封装的对象
                               batch_size=5,  # 输出的batch_size
                               shuffle=True,  # 随机输出
                               num_workers=0) # 只有一个进程

""" 训练输出 """
# 实例化一个类
net1 = linear_first(1)
net2 = linear_second(1)
net3 = linear_third(1)

# 定义优化器
optimizer1 = optim.SGD(net1.parameters(), lr=0.000001)
optimizer2 = optim.SGD(net2.parameters(), lr=0.000001)
optimizer3 = optim.SGD(net3.parameters(), lr=0.000001)

# loss函数的选择
cost = nn.MSELoss()

# 开始训练
epochs = 100
lloss1 = []
lloss2 = []
lloss3 = []
for epoch in range(epochs):
    for (x_train, y_train) in tensor_dataloader:
        # 计算输出
        output1 = net1(x_train)
        output2 = net2(x_train)
        output3 = net3(x_train)
        # 计算损失值
        loss1 = cost(output1, y_train)
        loss2 = cost(output2, y_train)
        loss3 = cost(output3, y_train)
        # 清零梯度缓存
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        # 计算梯度
        loss1.backward()
        loss2.backward()
        loss3.backward()
        # 更新参数
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
    lloss1.append(loss1)
    lloss2.append(loss2)
    lloss3.append(loss3)
    print('epoch1:', epoch, 'time loss1:', loss1)
    print('epoch2:', epoch, 'time loss2:', loss2)
    print('epoch3:', epoch, 'time loss3:', loss3)

# 可视化结果
plt.figure()
plt.plot(range(len(lloss1)), lloss1, c='k', label='one_net')
plt.plot(range(len(lloss2)), lloss2, c='b', label='two_net')
plt.plot(range(len(lloss3)), lloss3, c='r', label='three_net')
plt.xlabel('epoch', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.legend()
plt.show()