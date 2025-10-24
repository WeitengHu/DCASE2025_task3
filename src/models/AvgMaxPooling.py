import torch
import torch.nn as nn
import torch.nn.functional as F
class AvgMaxPooling2d(nn.Module):
    # DCASE2024 rank3使用的pooling方式  
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgMaxPooling2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        return avg_out + max_out  # 算术求和