#  Name : Weishuo
#  Time : 2022/4/6 12:56
# E-mail: muyeqingfeng@126.com

from torch import nn

class simpleModel(nn.Module):
	def __init__(self, input_dim, hidden1_n, hidden2_n, output_dim):
		super(simpleModel, self).__init__()

		# 构建网络
		self.Layer = nn.Sequential(# 第一层网络
									nn.Linear(input_dim, hidden1_n),
		                            nn.BatchNorm1d(hidden1_n),
		                            nn.ReLU(True),
									# 第二层网络
									nn.Linear(hidden1_n, hidden2_n),
									nn.BatchNorm1d(hidden2_n),
									nn.ReLU(True),
									# 输出层
									nn.Linear(hidden2_n, output_dim)
									)

	# 构建前向传播计算过程
	def forward(self, x):
		x = self.Layer(x)
		return x
