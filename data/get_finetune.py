import shutil
import time

import cv2
from util import SelectiveSearch
from util import parse_xml
from util import compute_ious
import numpy as np
from tqdm import tqdm
import os
from numpy import argsort
#载入图片并selectivesearch选出来加入文件
dst_root="./funetune/"
src_root="./deal_data/"
selectivesearch = SelectiveSearch(mode="q")
def get_list(src_xml,src_jpg):
    img = cv2.imread(src_jpg)
    #Selective Search
    rects = selectivesearch(img)
    #标注文件
    bbs=parse_xml(src_xml)

    if len(bbs)>0:
        iou_list=compute_ious(rects,bbs)
    else:
        return [],[]

    maxare=-1
    for bb in bbs:
        xmin,ymin,xmax,ymax=bb
        are = (xmax-xmin)*(ymax-ymin)
        if are>maxare:
            maxare=are

    positive_list=[]
    negative_list=[]
    #分类判断
    for i,iou in tqdm(enumerate(iou_list)):
        #获取坐标、计算大小
        xmin,ymin,xmax,ymax=rects[i]
        size=(xmax-xmin)*(ymax-ymin)
        if iou>0.5:#正样本
            positive_list.append((rects[i],iou))
        elif iou>0.1 and size>maxare/5.0:
            #负样本
            negative_list.append((rects[i],iou))
    if positive_list.__len__()<16 or negative_list.__len__()<48:
        return [],[]
    positive_list=sorted(positive_list,key=lambda t:1/(t[1]+0.1))
    negative_list=sorted(negative_list,key=lambda t:1/(t[1]+0.1))
    positive=[i[0] for i in positive_list]
    negative=[i[0] for i in negative_list]
    return positive[:16],negative[:48]

if __name__ == '__main__':
    for name in ['train','val']:
        positive_number=0
        negative_number=0

        src=os.path.join(src_root,name)
        dst=os.path.join(dst_root,name)
        dst_csv=os.path.join(dst,"car.csv")
        dst_annotation = os.path.join(dst,"Annotations")
        dst_jpeg = os.path.join(dst,"JPEGImages")
        src_csv=os.path.join(src,"car.csv")
        src_annotations = os.path.join(src, "Annotations")
        src_img = os.path.join(src, "JPEGImages")

        #读取目录
        samples = np.loadtxt(src_csv,dtype=np.str_)
        tlt=len(samples)
        car = []
        #将每个图片输入到SelectiveSearch中
        #将IOU>0.5设置为正样本
        for i,sample in tqdm(enumerate(samples)):
            since = time.time()
            src_xml=os.path.join(src_annotations,sample+".xml")
            src_jpg=os.path.join(src_img,sample+".jpg")

            positive_list,negative_list = get_list(src_xml,src_jpg)

            if len(positive_list)==0:
#                print("pass:",i)
                continue
            car.append(sample)
            positive_number+=len(positive_list)
            negative_number+=len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation, sample + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation, sample + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg, sample + '.jpg')
            # 保存图片
            shutil.copyfile(src_jpg, dst_jpeg_path)
            # 保存正负样本标注
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')
            time_elapsed = time.time() - since
#            print('{}parse {}.png in {:.0f}m {:.0f}s process: [{}/{}] {:.2f}%'.format(name,sample, time_elapsed // 60, time_elapsed % 60,i,tlt,(i/tlt)*100))
        #保存提取出来的编号列表
        np.savetxt(dst_csv, np.asarray(car),fmt='%s',delimiter=" ")

        print('%s positive num: %d' % (name, positive_number))
        print('%s negative num: %d' % (name, negative_number))
    print('done')
