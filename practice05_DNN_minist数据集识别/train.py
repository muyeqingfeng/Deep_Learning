#  Name : Weishuo
#  Time : 2022/4/6 16:20
# E-mail: muyeqingfeng@126.com

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net import simpleModel

#　初始化参数
batch_size = 100
learning_rate = 1e-2

# 实例化数据转换器
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# 获取数据
train_dataset = datasets.MNIST(root='.\data', train=True,
                               transform=data_transform, download=False)
test_dataset = datasets.MNIST(root='.\data', train=False,
                              transform=data_transform)

# 随机获取批量 训练数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型
model = simpleModel(28*28, 300, 600, 10)

# 选择成本函数
criterion = nn.CrossEntropyLoss()

# 选择优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 模型训练
lloss = []
for epoch in range(10):
	# 启用 BatchNormalization 和 Dropout 进行训练模式
	model.train()
	for data in train_loader:
		# 获取图像 及 标签
		img, label = data

		# 将数据维度由 [input_dim, 1, 28, 28] 转换为 [input_dim, 1*28*28]
		img = img.view(img.size(0), -1)
		img = Variable(img)
		label = Variable(label)

		# 计算输出
		output = model(img)
		# 计算损失值
		loss = criterion(output, label)
		# 清零梯度缓存
		optimizer.zero_grad()
		# 计算损失函数的梯度
		loss.backward()
		# 更新参数
		optimizer.step()

	print('epoch:{0}, loss:{1}'.format(epoch, loss))
	lloss.append(loss)

# 可视化
plt.plot(list(range(len(lloss))), lloss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 保存模型
torch.save(model, 'mnist_DNN')

