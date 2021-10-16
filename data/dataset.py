import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import util


class FastDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super(FastDataset, self).__init__()
        self.transform = transform

        samples = util.parse_car_csv(root_dir)
        jpeg_list = list()
        # 保存{'image_id': ?, 'positive': ?, 'bndbox': ?}
        box_list = list()
        for i in range(len(samples)):
            sample_name = samples[i]

            jpeg_path = os.path.join(root_dir, 'JPEGImages', sample_name + '.jpg')
            Annatations = os.path.join(root_dir, 'Annotations')
            bndbox_path = os.path.join(Annatations, sample_name+"_0" + '.csv')
            positive_path = os.path.join(Annatations, sample_name+ "_1" '.csv')
            bb = os.path.join(Annatations, sample_name+ "_-1" '.csv')

            jpeg_list.append(cv2.imread(jpeg_path))
            bndboxes = np.loadtxt(bndbox_path, dtype=np.int, delimiter=' ')
            positives = np.loadtxt(positive_path, dtype=np.int, delimiter=' ')
            bbs = np.loadtxt(bb, dtype=np.int, delimiter=' ')
            # print(len(bbs.shape))
            if len(bbs.shape)==1:
                #[n, 4]
                # print(bbs)
                bbs=np.expand_dims(bbs,0)
                # print(bbs)
            # print(positives.shape)#[16,4]
            #print(bbs.shape)#[n,4]
            bb=[]
            for id,b in enumerate(bbs):
                xmin, ymin, xmax, ymax = b
                g_w = xmax - xmin
                g_h = ymax - ymin
                g_x = xmin + g_w / 2
                g_y = ymin + g_h / 2
                # positive:torch.Size([16,4])
                lbb=[]
                for p in positives:
                    xmin, ymin, xmax, ymax = p
                    p_w = xmax - xmin
                    p_h = ymax - ymin
                    p_x = xmin + p_w / 2
                    p_y = ymin + p_h / 2
                    t_x = (g_x - p_x) / p_w
                    t_y = (g_y - p_y) / p_h
                    t_w = np.log(g_w / p_w)
                    t_h = np.log(g_h / p_h)
                    lbb.append([t_x, t_y, t_w, t_h])
                bb.append(lbb)
            bbs=np.asarray(bb)
            #print(bb.shape)          #(n, 16, 4)
            box_list.append({'image_id': i, 'posi': positives, 'neg': bndboxes,"bbs":bbs})
        self.jpeg_list = jpeg_list
        self.box_list = box_list
        self.samples=samples

    def __getitem__(self, index: int):

        box_dict = self.box_list[index]
        image_id = box_dict['image_id']
        positive = box_dict['posi']
        neg = box_dict['neg']
        bbs = box_dict['bbs']
        # 获取预测图像
        jpeg_img = self.jpeg_list[image_id]
        image = jpeg_img
        if self.transform:
            image = self.transform(image)
        return image, positive , neg, bbs

    def __len__(self):
        return len(self.box_list)

    def get_bndbox(self, bndboxes, positive):
        """
        返回和positive的IoU最大的标注边界框
        :param bndboxes: 大小为[N, 4]或者[4]
        :param positive: 大小为[4]
        :return: [4]
        """

        if len(bndboxes.shape) == 1:
            # 只有一个标注边界框，直接返回即可
            return bndboxes
        else:
            scores = util.iou(positive, bndboxes)
            return bndboxes[np.argmax(scores)]

