import torch
import torch.nn as nn

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5,1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )
        self.fcs = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    def forward(self,input):
        input = self.conv1(input)
        input = self.conv2(input)
        input = input.view(-1,16*4*4)
        input = self.fcs(input)
        return input
        