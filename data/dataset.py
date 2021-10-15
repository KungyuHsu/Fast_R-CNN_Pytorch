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
        print(type(image))
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

