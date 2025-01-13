import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMNISTBackbone(nn.Module):
    def __init__(self):
        super(TinyMNISTBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x28x28
        self.pool = nn.MaxPool2d(2, 2)  # Output: 64x14x14
        #self.dropout_conv = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 14 * 14, 64)
        self.fc3 = nn.Linear(64, 32)
        #self.dropout_fc = nn.Dropout(0.2)
        
        #self.bn1 = nn.BatchNorm2d(32)
        #self.bn2 = nn.BatchNorm2d(64)
        #self.bn_fc1 = nn.BatchNorm1d(256)
        #self.bn_fc2 = nn.BatchNorm1d(128)
        self.bn_fc3 = nn.BatchNorm1d(32)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #x = self.dropout_conv(x)
        
        x = self.flatten(x)
        
        #x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.fc1(x))
        #x = self.dropout_fc(x)
        #x = F.relu(self.bn_fc2(self.fc2(x)))
        #x = F.relu(self.fc2(x))
        #x = self.dropout_fc(x)
        x = self.bn_fc3(self.fc3(x))
        #x = self.fc3(x)
        return x