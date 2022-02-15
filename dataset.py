# -*- coding: utf-8 -*-
import os
import random

#data_path = 'badminton/001/' #要存放dataset的path 到VOC2007前一個資料夾

trainval_percent = 0.9 #訓練加驗證集佔全部資料集比例
train_percent = 0.9 # 訓練集佔訓練加驗證集比例
#xmlfilepath = data_path+'VOC2007/Annotations/' #標註檔路徑
#xmlfilepath = 'C:/Users/ricky/Desktop/gopro_annotation/1104_player01/ball11/' #標註檔路徑
#txtsavepath = data_path+'VOC2007/ImageSets/Main' #訓練、驗證、測試清單路徑
xmlfilepath = 'scaphoid\VOC2007\Annotations'
txtsavepath = 'scaphoid\VOC2007\ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open('scaphoid\VOC2007/ImageSets/Main/trainval.txt', 'w') #指定訓練加驗證集清單檔
ftest = open('scaphoid\VOC2007/ImageSets/Main/test.txt', 'w') #指定測試資料集清單
ftrain = open('scaphoid\VOC2007/ImageSets/Main/train.txt', 'w') #指定訓練資料集清單
fval = open('scaphoid\VOC2007/ImageSets/Main/val.txt', 'w') #指定驗證資料集清單

for i  in list:
    name=total_xml[i][:-5]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()