import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


#下面是resnet通用解构的定义
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(
        in_channels,  # 输入深度（通道）
        out_channels, # 输出深度
        kernel_size=3,# 滤波器（过滤器）大小为1*1
        stride=stride,# 步长，默认为1
        padding=1,    # 0填充一层
        bias=True    # 不设偏置
    )


class BasicBlock(nn.Module):
    expansion = 1 # 是对输出深度的倍乘，在这里等同于忽略

    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride) # 3*3卷积层
        # self.bn1 = nn.BatchNorm2d(out_channels) # 批标准化层
        self.relu = nn.ReLU(True) # 激活函数

        self.conv2 = conv3x3(out_channels,out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # 这个是shortcut的操作
        self.stride = stride # 得到步长

    def forward(self,x):
        residual = x # 获得上一层的输出

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None: # 当shortcut存在的时候
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, n_input_channels,num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.n_input_channels = n_input_channels
        self.conv1 = nn.Conv2d(self.n_input_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(49152, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # print("x.shape:",x.shape)
        x = x.view(x.size()[0], -1)
        # x = torch.flatten(x,1)
        x = self.fc(x) #shape=12800

        return x

def resnet18(**kwargs):
    model = ResNet(BasicBlock,[2,2,2,2],**kwargs)

    return model


# 下面部分是stable baseline3的自定义策略网络中extractor代码，
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.resnet18 = resnet18(
            n_input_channels = observation_space.shape[0],
            num_classes=3072
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.resnet18(observations)
