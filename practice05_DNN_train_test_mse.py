#  Name : Weishuo
#  Time : 2022/4/7 13:58
# E-mail: muyeqingfeng@126.com

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net import simpleModel

# 初始化参数
batch_size = 50
learning_rate = 2e-3

# 实例化数据转换器 ———— 标准化
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5,], [0.5,])])

# 获取数据 并 进行预处理
train_dataset = datasets.MNIST(root='..\data', train=True,
                               transform=data_transform, download=False)
test_dataset = datasets.MNIST(root='..\data', train=False,
                              transform=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10000)

# 实例化模型
model = simpleModel(28*28, 1000, 500, 50, 1)
# 选择成本函数
criterion = nn.MSELoss()
# 选择优化器
optimizer =  optim.SGD(model.parameters(), lr=learning_rate)

# 模型训练
lloss_train = []
lloss_test = []
acc_train_lst = []
acc_test_lst = []
for epoch in range(50):
    model.train()
    for data in train_dataloader:
        img, label = data
        img = img.view(img.size(0), -1)

        # 计算输出
        output = model(img)
        # 计算损失函数
        loss = criterion(output.squeeze(), label.float())
        # 清零梯度缓存
        optimizer.zero_grad()
        # 计算损失函数的梯度
        loss.backward()
        # 更新参数
        optimizer.step()

    print('epoch:{0}, loss:{1}'.format(epoch, loss))
    # 记录 训练集 损失函数
    lloss_train.append(loss)
    # 记录 训练集 正确率
    acc_train = (output.round().squeeze() == label).sum() / batch_size
    acc_train_lst.append(acc_train)

    # 测试集
    model.eval()
    with torch.no_grad():
        for data_test in test_dataloader:
            img_test, label_test = data_test
            img_test = img_test.view(img_test.size(0), -1)
            output_test = model(img_test)
            loss_test = criterion(output_test.squeeze(), label_test.float())
            acc_test = (output_test.round().squeeze() == label_test).sum() / len(output_test)
        lloss_test.append(loss_test)
        acc_test_lst.append(acc_test)


plt.figure(figsize=(12, 6))

# 训练误差
ax1 = plt.subplot(2, 2, 1)
ax1.plot(lloss_train)
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('train_loss(lr={0})'.format(eval(str(learning_rate))))

# 测试误差
ax2 = plt.subplot(2, 2, 2)
ax2.plot(lloss_test)
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('test_loss')

# 训练准确率
ax3 = plt.subplot(2, 2, 3)
ax3.plot(acc_train_lst)
ax3.set_xlabel('epoch')
ax3.set_ylabel('accuracy')
ax3.set_title('train_accuray')

# 测试准确率
ax4 = plt.subplot(2, 2, 4)
ax4.plot(acc_test_lst)
ax4.set_xlabel('epoch')
ax4.set_ylabel('accuracy')
ax4.set_title('test_accuray')

plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.tight_layout()
plt.show()


# 保存模型
# torch.save(model, 'minist_DNN_mse')