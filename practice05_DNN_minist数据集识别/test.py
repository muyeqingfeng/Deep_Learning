#  Name : Weishuo
#  Time : 2022/4/6 16:20
# E-mail: muyeqingfeng@126.com

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 模型导入
model=torch.load('mnist_DNN')

# 实例化一个数据转换器
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5,])])
# 验证数据集导入
test_dataset = datasets.MNIST(root='..\data', train=False, transform=data_transform)
test_loader = DataLoader(test_dataset)

# 模型验证
model.eval()
acc = 0
num_loss = 0
nnum_loss = []
criterion = nn.MSELoss()

def cost(y_pre, y):
	return (y_pre - y) ** 2

for data in test_loader:
	img, label = data
	img = img.view(img.size(0), -1)

	img = Variable(img)
	label = Variable(label)

	output = model(img)

	# 计算模型值 与 真实值之间的数值偏差
	num_loss += cost(output.argmax(axis=1), label) / len(test_dataset)
	nnum_loss.append(num_loss)
	# 统计预测正确的个数 及 正确率
	acc += ((output.argmax(axis=1) == label).sum()) / len(test_dataset)

print(acc)
print(num_loss)



