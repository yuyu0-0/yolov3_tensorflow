# -*- coding: utf-8 -*-
import os
import random
import glob
import cv2
from xml.dom.minidom import Document
import json as js
from os import listdir
#data_path = 'badminton/001/' #要存放dataset的path 到VOC2007前一個資料夾

testdata_percent = 1 #訓練加驗證集佔全部資料集比例


valtxt_file = '2007_val.txt'
fvaltxt = open(valtxt_file, 'w')
#json_filename = 'D:\\final_yolo\Demo/Annotations/'
json_filename = 'scaphoid/VOC2007/Annotations'
        
#convert_annotation(image_id, list_file)

#list_file = '2007_demo.txt'
datapath = 'scaphoid/VOC2007/JPEGImages/'
#imgname = 'D:\\final_yolo\Demo/Images/'
imgname = 'scaphoid/VOC2007/JPEGImages/'
#json_filename = 'D:\\final_yolo\Demo/Annotations/'
json_filename = 'scaphoid/VOC2007/Annotations/'
demo_filenametxt = 'scaphoid/VOC2007/ImageSets/Main/val.txt'
demo_json2txt = 'scaphoid/VOC2007/Annotations'

f = open('scaphoid/VOC2007/ImageSets/Main/val.txt')
for line in f.readlines():
    print(line)
    image_id = line.strip('\n')
    #image_id = line[:-1]
    listfile = imgname+image_id+'.bmp'
    def convert_annotation(image_id,bmp):
        with open('scaphoid\\VOC2007\\Annotations\\' + image_id +'.json' , newline='') as jsonfile:
            data = js.load(jsonfile)
            #print(data)
            b = data[0]["bbox"]
            print(b)
            for i in b:
                #print(i)
                print(str(i))
            ann = ",".join([str(i) for i in b])+ ',' + str('0')
            print(ann)
        
        name= bmp+" " + ann+'\n'
        print(name)
        #name = str(name)
        fvaltxt.write(name)
        print('ttttttttttttttttttttttttttttttttttttttttt')
    convert_annotation(image_id,listfile)
f.close
print('00000000000000000000000000000000000')





