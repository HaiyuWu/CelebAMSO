from torchvision import models
import torch.nn as nn


def conv2d(in_feature, out_feature, kernal_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_feature, out_feature, kernal_size, stride, padding),
                         nn.BatchNorm2d(out_feature),
                         nn.ReLU(True))


class Moon(nn.Module):
    def __init__(self):
        super(Moon, self).__init__()
        self.layers = nn.Sequential(conv2d(3, 64, 3, 1, 1),
                                    conv2d(64, 64, 3, 1, 1),
                                    nn.MaxPool2d(2, 2),
                                    conv2d(64, 128, 3, 1, 1),
                                    conv2d(128, 128, 3, 1, 1),
                                    nn.MaxPool2d(2, 2),
                                    conv2d(128, 256, 3, 1, 1),
                                    conv2d(256, 256, 3, 1, 1),
                                    conv2d(256, 256, 3, 1, 1),
                                    nn.MaxPool2d(2, 2),
                                    conv2d(256, 512, 3, 1, 1),
                                    conv2d(512, 512, 3, 1, 1),
                                    conv2d(512, 512, 3, 1, 1),
                                    nn.MaxPool2d(2, 2),
                                    conv2d(512, 512, 3, 1, 1),
                                    conv2d(512, 512, 3, 1, 1),
                                    conv2d(512, 512, 3, 1, 1),
                                    nn.MaxPool2d(2, 2),
                                    nn.AdaptiveAvgPool2d(7))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(25088, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 1000),
                                        nn.ReLU(True),
                                        nn.Dropout(0.5),
                                        # nn.Linear(1000, 40)
                                        nn.Linear(1000, 1)
                                        )

    def forward(self, x):
        x = self.layers(x)
        out = self.classifier(x)
        return out
