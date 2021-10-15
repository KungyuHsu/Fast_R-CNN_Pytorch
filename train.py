import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Fast_RCNN import Fast_RCNN
import torchvision.transforms as transforms
import torch.optim as optim
from data.dataset import FastDataset
from data.finetune_sample import CustomBatchSampler
import time
import copy
from torch.nn import functional as F
from config import *
def load_data(data_root_dir):
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}

    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)
        #定义Dataset
        data_set = FastDataset(data_dir, transform=transform)
        #定义加载器
        data_loader = DataLoader(data_set, batch_size=1, num_workers=4, drop_last=True)

        data_loaders[name] = data_loader

    return data_loaders
def gpu(input):
    res=[i.to(device) for i in input]
    return res
def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    # best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        torch.cuda.empty_cache()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            batchtime=time.time()
            #batchNormal
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            optimizer.zero_grad()
            # Iterate over data.
            loss=torch.Tensor([0]).to(device)
            for id,data in enumerate(data_loaders[phase]):
                (image, positive, neg, bbs)=data
                image=image.to(device)
                positive=positive.to(device)
                neg=neg.to(device)
                bbs=bbs.to(device)
                lo=len(data_loaders[phase])
                # zero the parameter gradients
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #positive
                    cls,t = model(image,positive)
                    lable=torch.ones(16,dtype=torch.int64).to(device)
                    loss += criterion(cls,lable,t,bbs)
                    pred = torch.argmax(cls,dim=1)
                    running_corrects += torch.sum(pred == 1)

                    cls,t = model(image,neg)
                    lable = torch.zeros(48,dtype=torch.int64).to(device)
                    loss += criterion(cls,lable,t,bbs)
                    pred = torch.argmax(cls,dim=1)
                    running_corrects += torch.sum(pred == 0)
                    # backward + optimize only if in training phase
                    if phase == 'train' and (id+1)%2==0:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                # statistics
                running_loss += loss.item() * image.__len__()
                if id%100==0:
                    print("process:[{} / {}] ".format(id,lo))
            if phase == 'train':
                # torch.cuda.empty_cache()
                lr_scheduler.step()
                torch.save(model.state_dict(), 'alexnet_car.pth')
            epoch_loss = running_loss / 128
            epoch_acc = running_corrects / 128
            # torch.cuda.empty_cache()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            bathcend=time.time()-batchtime
            print('Using time: {:.4f}'.format(bathcend))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_weights = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return model


def smooth_loss(t,bbs):
    """
    t:回归框[4]
    v:标注框
    """
    bb=bbs[0]
    x = t - bb
    x = torch.sum(x, 1)
    loss=x.unsqueeze(1)
    for bb in bbs[1:]:
        x=t-bb
        x=torch.sum(x,1).unsqueeze(1)
        loss=torch.cat((loss,x),1)
    x=torch.min(loss,dim=1)[0]
    loss=0
    for xi in x:
        if xi<1 and xi>-1:
            loss+=torch.sum(0.5*xi*xi)
        else:
            loss+=torch.sum(abs(xi)-0.5)
    return loss

def mutyloss(p,u,t,v,lamda=1):
    """
    p:每一个类别的概率
    u:正确之类别,0,1,2···
    t:回归框
    v:标注框
    """
    Lcls=F.cross_entropy(p,u)
    ti=t[u[0]]
    if u[0]>0:
        Lloc=smooth_loss(ti,v)
    else:
        Lloc=0
    return Lcls+lamda*Lloc

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    model = Fast_RCNN()
    data_loaders = load_data(r'./data/funetune')

    model = model.to(device)
    # 定义交叉损失
    criterion = mutyloss
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # 动态学习率，不过原论文直接定义1e-3
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # 训练模型和保存
    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=25)
    # 保存最好的模型参数
    torch.save(best_model.state_dict(), 'alexnet_car.pth')
