import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from model import lenet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 定义模型、损失函数和优化器
model = lenet.Lenet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
def train(pre_epochs,epochs):
    losses = []
    #如果预训练过了就加载已有权重
    if pre_epochs >= 1:
        model.load_state_dict(torch.load('saved_model.pth'))
    #迭代训练
    for epoch in range(pre_epochs+1, epochs):
        for i,(images,labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, 10, i+1, len(train_loader), loss.item()))
            losses.append(loss.item())
        torch.save(model.state_dict(), 'saved_model.pth') #每一个epoch训练结束保存一次模型
    #损失函数可视化
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
                
if __name__=='__main__':
    train(1,10)
    