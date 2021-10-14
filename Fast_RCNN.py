import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import roi_pool
from selective_search import SelectiveSearch

class Fast_RCNN(nn.Module):

    def __init__(self,class_num=1):
        super(Fast_RCNN, self).__init__()
        self.feature = models.alexnet(pretrained=True).features
        self.ss=SelectiveSearch()
        self.FC=nn.Sequential(
            nn.Linear(256*7*7,4096),
            nn.Linear(4096,4096)
        )
        self.bbs=nn.Linear(4096,(class_num+1)*4)
        self.cls=nn.Linear(4096,class_num+1)

    def forward(self,X,bbs):
        """
        X
        """
        x=self.feature(X)
        rois=roi_pool(x,bbs,7,13/227)
        #Tensor[K, C, output_size[0], output_size[1]]
        rois = torch.flatten(rois, 1)
        #rois = [K,256*7*7]
        x=self.FC(rois)
        #x = [K,4096]
        cls = self.cls(x)
        bbs = self.bbs(x).reshape([-1,4])
        return cls,bbs
    """
    cls:分类情况 
    bbs:回归框
    """
