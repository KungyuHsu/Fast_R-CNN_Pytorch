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
        data_loader = DataLoader(data_set, batch_size=2, num_workers=8, drop_last=True)

        data_loaders[name] = data_loader

    return data_loaders

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
            for id,(image, positive , neg,  bbs) in enumerate(data_loaders[phase]):
                lo=len(data_loaders[phase])
                image=image.to(device)
                positive=positive.to(device)
                neg=neg.to(device)
                bbs=bbs.to(device)
                # zero the parameter gradients
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #positive
                    cls,t = model(image,positive)

                    loss = criterion(cls,1,t,bbs)
                    _,pred = torch.max(cls)
                    running_corrects += torch.sum(pred == 1)

                    cls,t = model(image,neg)

                    loss += criterion(cls,0,t,bbs)
                    _,pred = torch.max(cls)
                    running_corrects += torch.sum(pred == 0)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                # statistics
                running_loss += loss.item() * image.size(0)
                if id%100==0:
                    print("process:[{} / {} ".format(id,lo))
            if phase == 'train':
                torch.cuda.empty_cache()
                lr_scheduler.step()
                torch.save(model.state_dict(), 'alexnet_car.pth')
            epoch_loss = running_loss / 128
            epoch_acc = running_corrects.double() / 128
            torch.cuda.empty_cache()
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
    model.load_state_dict(best_model_weights)
    return model


def smooth_loss(x):
    """
    t:回归框[4]
    v:标注框
    """
    if x<1:
        return torch.sum(0.5*x^2)
    else:
        return torch.sum(abs(x)-0.5)

def mutyloss(p,u,t,v,lamda=1):
    """
    p:每一个类别的概率
    u:正确之类别,0,1,2···
    t:回归框
    v:标注框
    """
    Lcls=F.cross_entropy(p,u)
    t_=t[u]
    if t_>0:
        Lloc=smooth_loss(t_-v)
    else:
        Lloc=0
    return Lcls+lamda*Lloc

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
