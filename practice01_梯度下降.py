#  Name : Weishuo
#  Time : 2022/3/30 15:33
# E-mail: muyeqingfeng@126.com

"""
问题：对于一个线性公式的问题，这个问题被建模成y=wx+b的问题，利用梯度下降算法，把最优w和b求出来
"""

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2.1, 4.1, 5.9, 7.8, 9.9]

def forward(x, w=0):
	return x * w

def cost(x, y, w=0):
	cost = 0
	for x_data, y_data in zip(x, y):
		y_predict = forward(x_data, w)
		cost += (y_predict - y_data) ** 2
	return cost / len(x)

def gradient(x, y, w=0):
	gra = 0
	for x_data, y_data in zip(x, y):
		gra += 2 * (forward(x_data, w) - y_data) * x_data
	return gra / len(x)

def main(loop=1000, step=0.001, w=0):
	loss = []
	for i in range(loop):
		loss0 = cost(x, y, w)
		w -= step * gradient(x, y, w)
		loss.append(loss0)
		print('epoch:', i, 'w=', w, 'loss=', loss0)
	return loss

if __name__ == '__main__':
	loss = main()
	plt.plot(range(1000), loss)
	plt.show()

