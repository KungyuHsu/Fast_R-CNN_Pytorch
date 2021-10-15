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
        self.class_num=class_num
        for p in self.feature.parameters():
            p.requires_grad=False
    def forward(self,X,bbs):
        """
        X
        """
        x=self.feature(X[0].unsqueeze(0))
        F=x
        for i in X[1:]:
            x=self.feature(i.unsqueeze(0))
            F=torch.cat((F,x),dim=0)
        print(F.shape)
        #bbs:torch.Size([2,16,4])
        #x:torch.Size([2,255,333,444])
        b = bbs[0].to("cuda")
        p = (torch.ones(b.shape[0]) * 0).unsqueeze(1).to("cuda")
        print(p.shape,b.shape)
        bb = torch.cat((p, b), 1)
        for i in range(self.class_num):
            i+=1
            b = bbs[i].to("cuda")
            p = (torch.ones(b.shape[0])*i).unsqueeze(1).to("cuda")
            b = torch.cat((p, b), 1)
            bb = torch.cat((bb,b),0)

        #b:Tensor[K, 5] 第一列为 0 or 1
        rois = roi_pool(input=F, boxes=bb, output_size=7, spatial_scale=13 / 227)
        #Tensor[K, C, output_size[0], output_size[1]]
        rois = torch.flatten(rois, 1)
        #rois = [K,256*7*7]
        x=self.FC(rois)
        #x = [K,4096]
        cls = self.cls(x)
        bbs = self.bbs(x).reshape([self.class_num+1,-1,4])
        return cls,bbs
    """
    cls:分类情况 
    bbs:回归框
    """
