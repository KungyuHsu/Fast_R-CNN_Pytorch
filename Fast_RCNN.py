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
        #torch.Size([2, 3, 333, 500])????

        rois=[]
        for i,x_ in enumerate(x):
            b=bbs[i]
            p = torch.ones(b.shape[0]).unsqueeze(1).to("cuda")
            b=torch.cat((p,b),1)
            print("*********")
            print(x_.shape)
            print("=========")
            print(b.shape)
            print(b)
            roi=roi_pool(x_,b,7,13/227)
            rois.append(roi)
        #Tensor[K, C, output_size[0], output_size[1]]
        rois=torch.Tensor(rois)
        print(rois.shape)
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
