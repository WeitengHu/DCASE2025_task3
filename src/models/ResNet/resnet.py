import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AvgMaxPooling import AvgMaxPooling2d
# def init_layer(layer):
#     """Initialize a Linear or Convolutional layer. """
#     nn.init.xavier_uniform_(layer.weight)
 
#     if hasattr(layer, 'bias'):
#         if layer.bias is not None:
#             layer.bias.data.fill_(0.)
            
    
# def init_bn(bn):
#     """Initialize a Batchnorm layer. """
#     bn.bias.data.fill_(0.)
#     bn.weight.data.fill_(1.)

def conv3x3(in_plane,out_plane,stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_plane,out_plane,kernel_size=3,stride=stride,padding=1,bias=False)

def conv1x1(in_plane,out_plane,stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_plane,out_plane,kernel_size=1,stride=stride,bias=False)




class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_planes,planes,stride=1,downsample=None,norm_layer=nn.BatchNorm2d):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_planes,planes,stride=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes,stride=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # self.use_pooling = use_pooling

        # # 如果需要下采样，在残差连接后使用 T-F Pooling
        # if use_pooling and stride == 2:
        #     # 时频联合池化：时间和频率方向都下采样
        #     self.tf_pool = AvgMaxPooling2d(kernel_size=stride, stride=stride)
        # else:
        #     self.tf_pool = None

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # # 在残差连接之后进行池化
        # if self.tf_pool is not None:
        #     out = self.tf_pool(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride=1,downsample=None,norm_layer=nn.BatchNorm2d):
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(in_planes,planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes,planes,stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes,planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        identity = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000,zero_init_residual=False,norm_layer=nn.BatchNorm2d,include_top=True,base_channels=64,in_channels=2):
        super(ResNet,self).__init__()
        self.in_planes = base_channels
        self.conv1 = nn.Conv2d(in_channels,base_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = norm_layer(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.tf_pool = AvgMaxPooling2d(kernel_size=(2,2), stride=(2,2))
        self.layer1 = self._make_layer(block,base_channels,layers[0],norm_layer=norm_layer)
        self.layer2 = self._make_layer(block,base_channels*2,layers[1],stride=2,norm_layer=norm_layer)
        self.layer3 = self._make_layer(block,base_channels*4,layers[2],stride=2,norm_layer=norm_layer)
        self.layer4 = self._make_layer(block,base_channels*8,layers[3],stride=2,norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(base_channels*8*block.expansion,num_classes)
        # self.freq_pool = AvgMaxPooling2d(kernel_size=(1,2), stride=(1,2))
        self.out_dim = base_channels*8 * block.expansion
        self.include_top = include_top
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def _make_layer(self,block,planes,num_blocks,stride=1,norm_layer=nn.BatchNorm2d):
        downsample = None
        # 当需要下采样时，downsample 分支也不使用 stride，而是依赖后续的 pooling
        if stride != 1 or self.in_planes != planes * block.expansion:
            # downsample 分支使用 stride=1 的卷积，pooling 在 block 内部的残差连接后进行
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes,planes,stride,downsample,norm_layer))

        self.in_planes = planes * block.expansion
        for i in range(1,num_blocks):
            layers.append(block(self.in_planes,planes))

        if stride != 1:
            # 在每个 layer 结束后进行时频联合池化
            layers.append(AvgMaxPooling2d(kernel_size=(2,2), stride=(2,2)))

        return nn.Sequential(*layers)

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet18(**kwargs):
    model = ResNet(BasicBlock,[2,2,2,2],**kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(**kwargs):
    model = ResNet(BasicBlock,[3,4,6,3],**kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck,[3,4,6,3],**kwargs)
    return model

def resnet101(**kwargs):
    model = ResNet(Bottleneck,[3,4,23,3],**kwargs)
    return model

def resnet152(**kwargs):
    model = ResNet(Bottleneck,[3,8,36,3],**kwargs)
    return model