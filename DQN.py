import torch
import torch.nn as nn
import torch.nn.functional as F



class DQN1(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.Conv1 = nn.Conv2d(4,4*8,8,stride=4)
        self.Conv2 = nn.Conv2d(4*8,4*8*2,4,stride=2)
        self.Conv3 = nn.Conv2d(64,64,3,stride=1)
        self.Linear1 = nn.Linear(3136,512)
        self.Linear2 = nn.Linear(512,6)

    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        x = torch.flatten(x,1,3)
        x = F.relu(self.Linear1(x))
        x = self.Linear2(x)
        return x


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.Conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.Conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.Linear1 = nn.Linear(self.feature_size(input_shape), 512)
        self.Linear2 = nn.Linear(512, num_actions)


    def feature_size(self, input_shape):
        return self.Conv3(self.Conv2(self.Conv1(torch.zeros(input_shape)))).view(1, -1).size(1)


    def forward(self, x): 
        x = torch.relu(self.Conv1(x))
        x = torch.relu(self.Conv2(x))
        x = torch.relu(self.Conv3(x))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.Linear1(x))
        return self.Linear2(x)
