#  Name : Weishuo
#  Time : 2022/4/8 14:19
# E-mail: muyeqingfeng@126.com

from torch import nn

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()

		# 提取特征层
		self.features = nn.Sequential(
			# 卷积层
			# 输入图像通道为1，因为使用的是黑白图，单通道的
			# 输出通道为32（代表使用32个卷积核），一个卷积和产生一个单通道的特征图
			# 卷积核kernel_size的尺寸为 3*3， stride代表每次卷积核移动像素个数为1
			# padding填充，为1代表图像长宽都多了两个像素
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
			# 批量归一化，跟上一层的out_channels大小相等，
			nn.BatchNorm2d(num_features=32),  # 28*28*32 ---> 28*28*32

			# 激活函数，inplace=true代表直接进行运算
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # ((28-3+2*1)/1)+1
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),

			# 最大池化层
			# kernel_size 为2*2的滑动窗口
			# stride为2，表示每次滑动距离为2个像素
			# 经过这一步，图像的大小变为1/4，即 28*28 ---> 7*7
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		# 分类层
		self.classifier = nn.Sequential(
			# Dropout层
			# p = 0.5代表该层的每个权重有0.5的可能性为0
			nn.Dropout(p=0.5),
			# 这里是通道数 4* 图像大小 7*7，然后输入到512个神经元中
			nn.Linear(64 * 7 * 7, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(512, 10)
		)

	def forward(self, x):
		# 　经过特征提取层
		x = self.features(x)
		# 输出结果必须 展开为 一维向量
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x