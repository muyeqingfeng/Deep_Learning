#  Name : Weishuo
#  Time : 2022/3/31 9:39
# E-mail: muyeqingfeng@126.com

"""
1.利用torch.utils.data.Dataset做数据的数据生成器
2.利用DataLoader做数据的载入，和打乱等操作。打乱是对index的打乱.
"""

# 利用pytorch生成数据
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class TensorDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return len(self.x)

x_data = np.random.randint(1,100,200)
y_data = 2 * x_data + np.random.randint(-20,20,200)
# x_data = np.random.randn(100)
# y_data = 2 * x_data
tensor_dataset = TensorDataset(x_data, y_data)

"""
打印测试
print(tensor_dataset[1])
print(len(tensor_dataset))
"""

tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=5,     # 输出的batch size
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程
"""# 以for循环形式输出
for data, target in tensor_dataloader:
	print(data, target)
	plt.scatter(data, target)
	plt.show()"""

def forward(x, w=0):
	return x * w

def cost(x, y, w=0):
	cost = 0
	if len(x) == 1:
		y_predict = forward(x[0], w)
		cost += (y_predict - y[0]) ** 2
	else:
		for x_data, y_data in zip(x, y):
			y_predict = forward(x_data, w)
			cost += (y_predict - y_data) ** 2
	return cost / len(x)

def gradient(x, y, w=0):
	gra = 0
	for x_data, y_data in zip(x, y):
		gra += 2 * (forward(x_data, w) - y_data) * x_data
	return gra / len(x)

def main(tensor_dataloader, loop=100, step=0.0000001, w=0):
	loss = []
	for i in range(loop):
		for (x, y) in tensor_dataloader:
			loss0 = cost(x, y, w)
			w -= step * gradient(x, y, w)
		loss.append(loss0)
		print('epoch:', i, 'w=', w, 'loss=', loss0)
	return loss, w

if __name__ == '__main__':
	loss,w = main(tensor_dataloader)

	plt.subplot(211)
	plt.plot(list(range(len(loss))), loss, label='GD')
	plt.xlabel('epoch')
	plt.ylabel('loss')

	plt.subplots_adjust(wspace=0, hspace=0.3)
	plt.subplot(212)
	plt.scatter(x_data, y_data, c='k', label='origin')
	plt.plot(x_data, w*x_data, c='r', linewidth=1.5 ,label='SGD')
	plt.xlabel('x_data')
	plt.ylabel('y_data')
	plt.legend()
	plt.show()



