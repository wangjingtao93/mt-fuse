import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射变换：y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批大小维度的其余维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

net_state_dict = net.state_dict()
for name, param in net_state_dict.items():
    print(name)


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))


output = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1,-1) # make it the same shape as output

criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn) # MSE_Loss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # Relu

net.zero_grad()     # 清零所有参数（parameter）的梯度缓存

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


import  torch.optim as optim

# 创建优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练中迭代
optimizer.zero_grad() # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # 更新参数



