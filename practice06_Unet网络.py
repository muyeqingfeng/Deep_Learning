#  Name : Weishuo
#  Time : 2022/4/13 14:16
# E-mail: muyeqingfeng@126.com

import torch
import torch.nn as nn
import torch.nn.functional as F

""" 下采样 """
class double_conv2d(nn.Module):
	def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
		super(double_conv2d, self).__init__()
		self.convlayer = nn.Sequential(# 第一个卷积层
									   nn.Conv2d(input_channel, output_channel,
		                                         kernel_size=kernel_size,
		                                         stride=stride, padding=padding, bias=True),
		                               nn.BatchNorm2d(output_channel),
		                               nn.ReLU(True),
									   # 第二个卷积层
									   nn.Conv2d(output_channel, output_channel,
									             kernel_size=kernel_size,
									             stride=stride, padding=padding, bias=True),
									   nn.BatchNorm2d(output_channel),
									   nn.ReLU(True)
									   )

	def forward(self, x):
		out = self.convlayer(x)
		return out

""" 上采样 """
class deconv2d(nn.Module):
	def __init__(self, input_channel, output_channel, kernel_size=2, stride=2):
		super(deconv2d, self).__init__()
		self.deconvlayer = nn.Sequential(
										nn.ConvTranspose2d(input_channel, output_channel,
										                   kernel_size=kernel_size,
										                   stride=stride, bias=True),
										nn.BatchNorm2d(output_channel),
										nn.ReLU(True))

	def forward(self, x):
		out = self.deconvlayer(x)
		return out

""" Unet模型 """
class Unet(nn.Module):
	def __init__(self):
		super(Unet, self).__init__()
		self.layer1_conv = double_conv2d(1, 64)
		self.layer2_conv = double_conv2d(64, 128)
		self.layer3_conv = double_conv2d(128, 256)
		self.layer4_conv = double_conv2d(256, 512)
		self.layer5_conv = double_conv2d(512, 1024)

		self.layer6_conv = double_conv2d(1024, 512)
		self.layer7_conv = double_conv2d(512, 256)
		self.layer8_conv = double_conv2d(256, 128)
		self.layer9_conv = double_conv2d(128, 64)
		self.layer10_conv = nn.Conv2d(64, 1, kernel_size=3,
		                              stride=1, padding=1, bias=True)


		self.deconv1 = deconv2d(1024, 512)
		self.deconv2 = deconv2d(512, 256)
		self.deconv3 = deconv2d(256, 128)
		self.deconv4 = deconv2d(128, 64)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		conv1 = self.layer1_conv(x)
		pool1 = F.max_pool2d(conv1, 2)

		conv2 = self.layer2_conv(pool1)
		pool2 = F.max_pool2d(conv2, 2)

		conv3 = self.layer3_conv(pool2)
		pool3 = F.max_pool2d(conv3, 2)

		conv4 = self.layer4_conv(pool3)
		pool4 = F.max_pool2d(conv4, 2)

		conv5 = self.layer5_conv(pool4)

		convt1 = self.deconv1(conv5)
		concat1 = torch.cat([convt1, conv4], dim=1)
		conv6 = self.layer6_conv(concat1)

		convt2 = self.deconv2(conv6)
		concat2 = torch.cat([convt2, conv3], dim=1)
		conv7 = self.layer7_conv(concat2)

		convt3 = self.deconv3(conv7)
		concat3 = torch.cat([convt3, conv2], dim=1)
		conv8 = self.layer8_conv(concat3)

		convt4 = self.deconv4(conv8)
		concat4 = torch.cat([convt4, conv1], dim=1)
		conv9 = self.layer9_conv(concat4)

		outp = self.layer10_conv(conv9)
		outp = self.sigmoid(outp)

		return outp


""" 测试 """
model = Unet()
inp = torch.randn(10, 1, 224, 224)
outp = model(inp)
print(outp.shape)