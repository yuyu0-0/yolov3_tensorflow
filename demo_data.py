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

#xmlfilepath = 'scaphoid\VOC2007\Annotations'
#txtsavepath = 'scaphoid\VOC2007\ImageSets\Main'
image_id = '04504012-R-91F-AP0.json'
demotxt_file = '2007_demo.txt'
fdemotxt = open(demotxt_file, 'w')
#json_filename = 'D:\\final_yolo\Demo/Annotations/'
json_filename = 'Demo/Annotations/'
        
#convert_annotation(image_id, list_file)

list_file = '2007_demo.txt'
datapath = 'Demo/Images/'
#imgname = 'D:\\final_yolo\Demo/Images/'
imgname = 'Demo/Images/'
#json_filename = 'D:\\final_yolo\Demo/Annotations/'
json_filename = 'Demo/Annotations/'
demo_filenametxt = 'Demo/ImageSets/Main/demo.txt'
demo_json2txt = 'Demo/Annotations'

fdemo = open(list_file, 'w') #指定訓練加驗證集清單檔
#fdemo_json2txt = open(demo_json2txt, 'w') #指定訓練加驗證集清單檔
files = os.listdir(datapath)
num=len(files)
list=range(num)
for i  in list:
    name=files[i]+'\n'
    jsonfilename = files[i][:-4]+'.json'
    bmpname = files[i][:-4]+'.bmp'
    #print(image_id)
    #print('ooooooooooooooooo')
   
    json_file = json_filename+image_id
    
    
    
    #file_name = str(imgname+files[i])+' '+ann+'\n'
    
    print(json_filename+jsonfilename)
    print(imgname+bmpname)
    listfile = imgname+bmpname
    def convert_annotation(image_id,bmp):
        with open('Demo\\Annotations\\' + image_id , newline='') as jsonfile:
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
        fdemotxt.write(name)
        print('ttttttttttttttttttttttttttttttttttttttttt')
    convert_annotation(jsonfilename,listfile)
    #print(files[i])
    #print(json_file)
    #fdemo.write(file_name)
fdemotxt.close

#fdemo.close()



#convert_annotation(image_id, list_file,cls_id)'''