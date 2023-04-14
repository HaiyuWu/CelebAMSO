import torch.nn as nn


def conv1(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, stride=2, padding=3),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU(True),
                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class Blocks(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsampling=False, expansion=4):
        super(Blocks, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels=out_channel, out_channels=out_channel*expansion, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channel*expansion),
                                   )
        self.downsampling = downsampling
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel*expansion, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_channel*expansion),
                                            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        res = x
        out = self.block(x)

        if self.downsampling:
            res = self.downsample(x)
        out += res
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = conv1(in_channel=3, out_channel=64)
        self.block1 = self.block_maker(in_channel=64, out_channel=64, stride=1, block_num=num_blocks[0])
        self.block2 = self.block_maker(in_channel=256, out_channel=128, stride=2, block_num=num_blocks[1])
        self.block3 = self.block_maker(in_channel=512, out_channel=256, stride=2, block_num=num_blocks[2])
        self.block4 = self.block_maker(in_channel=1024, out_channel=512, stride=2, block_num=num_blocks[3])

        self.average = nn.AdaptiveAvgPool2d(1)
        # self.average = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def block_maker(self, in_channel, out_channel, stride, block_num):
        layers = []
        layers.append(Blocks(in_channel, out_channel, stride, downsampling=True))
        for i in range(1, block_num):
            layers.append(Blocks(out_channel*self.expansion, out_channel))
        return nn.Sequential(*layers)

    def forward(self, image):
        x = self.conv1(image)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.average(x)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


def ResNet51():
    return ResNet([3, 4, 6, 3])


if __name__ == '__main__':
    from torchsummary import summary
    model = ResNet51().cuda()
    summary(model, (3, 224, 224))
