import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        model = models.alexnet(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
