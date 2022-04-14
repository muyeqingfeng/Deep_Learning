#  Name : Weishuo
#  Time : 2022/4/14 16:40
# E-mail: muyeqingfeng@126.com

import torch.nn as nn
import torch.functional as F

""" 实现残差模块 """
class Block(nn.Module):
	def __init__(self, input_channel, output_channel, kerner_size=3, strides=1, same_shape=True):
		super(Block, self).__init__()
		self.same_shape = same_shape

		if not same_shape:
			strides = 2
		self.strides = strides

		self.block = nn.Sequential(nn.Conv2d(input_channel, output_channel,
								             kernel_size=7, stride=strides,
								             padding=3, bias=False),
								   nn.BatchNorm2d(output_channel),
								   nn.ReLU(True),

								   nn.Conv2d(output_channel, output_channel,
								             kernel_size=3, padding=1,
								             bias=False),
								   nn.BatchNorm2d(output_channel)
		                          )

		if not same_shape:
			self.convlayer = nn.Sequential(nn.Conv2d(input_channel, output_channel,
			                                         kernel_size=1, stride=strides,
			                                         padding=1, bias=False),
			                               nn.BatchNorm2d(output_channel))

	def forward(self, x):
		out = self.block(x)
		if not self.same_shape:
			x = self.convlayer(x)
		return F.relu(out + x)

""" 开始实现 ResNets34 """
class ResNet34(nn.Module):
	def __init__(self, num_classes=10):
		super(ResNet34, self).__init__()
		# 最开始的几层
		self.pre_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
		                                         kernel_size=7, stride=2,
		                                         padding=1, bias=False),
		                               nn.BatchNorm2d(64),
		                               nn.ReLU(True),
		                               nn.MaxPool2d(kernel_size=3, stride=2,
		                                            padding=1))
		# 分别有3,4,6,3个残差块
		self.layer1 = self._make_layer(64, 64, block_nums=3)
		self.layer2 = self._make_layer(128, 128, block_nums=4, stride=2)
		self.layer3 = self._make_layer(256, 256, block_nums=6, stride=2)
		self.layer4 = self._make_layer(512, 512, block_nums=3, stride=2)

		# 分类用的全连接
		self.fc = nn.Linear(512, num_classes)

	def _make_layer(self, input_channels, output_channels, block_nums, stride=1):
		layers = []
		if stride !=1:
			layers.append(Block(input_channels, output_channels, stride, same_shape=False))
		else:
			layers.append(Block(input_channels, output_channels, stride))

		for i in range(1, block_nums):
			layers.append(Block(output_channels, output_channels))
		return nn.Sequential(*layers)


	def forward(self, x):
		x = self.pre_layer(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = F.avg_pool2d(x, 7)
		x = x.view(x.size(0), -1)
		return self.fc(x)

