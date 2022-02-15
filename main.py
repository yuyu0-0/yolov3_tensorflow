from PyQt5 import QtWidgets, QtGui, QtCore

from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
from test_path import Ui_Dialog
import sys
import os
import cv2
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
from itertools import product
import math
import glob
import demo_data
from PyQt5.QtGui import QImage, QPixmap
import shutil
import subprocess
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.label.setScaledContents(True)
        self.ui.label_2.setScaledContents(True)
        self.ui.pushButton.clicked.connect(self.select_folder)
        self.ui.pushButton_2.clicked.connect(self.pre)
        self.ui.horizontalSlider.valueChanged.connect(self.valueChange)
        
    def select_folder(self):
        global data_img_path
        dir_ = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select project folder:', './', QtWidgets.QFileDialog.ShowDirsOnly)
        print(dir_)
        dirpath = str(dir_)
        print(dirpath) 
        
        
        source_folder = dirpath+"/Annotations/Scaphoid_Slice/"
        destination_folder = "Demo\\Annotations\\"
        
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + file_name            
            file_name = file_name.replace(' ', '-')
            destination = destination_folder + file_name
            print(file_name)
            print('kkkkkkkkkkkkkkkkkkkkk')  
            print(source)  
            if os.path.isfile(source):
                shutil.copy(source, destination) 
                print(destination)
            else:
                print('uuuuuuuuuuuuuuuuuuuuuuuuuu')
        
      
        source_folder = dirpath+"/Images/Fracture/"
        destination_folder = "Demo\\Images\\"
        
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + file_name            
            file_name = file_name.replace(' ', '-')
            destination = destination_folder + file_name
            print(file_name)
            print('777777777777')  
            print(source)  
            if os.path.isfile(source):
                shutil.copy(source, destination) 
                print(destination)
            else:
                print('5555555555555555')
        
        source_folder = dirpath+"/Images/Normal/"
        destination_folder = "Demo\\Images\\"
        
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + file_name            
            file_name = file_name.replace(' ', '-')
            destination = destination_folder + file_name
            print(file_name)
            print('66666666666666')  
            print(source)  
            if os.path.isfile(source):
                shutil.copy(source, destination) 
                print(destination)
            else:
                print('888888888888')
    def valueChange(self):
        global data_img_path
        pathDir = os.listdir('Demo/Images')
        paths = []
        for path in pathDir:
            paths.append('./Demo/Images/'+path)
        scaphoid_img_path = paths
        self.ui.horizontalSlider.setMaximum(len(pathDir)-1)
        pos= self.ui.horizontalSlider.value()
        
        
        result_pathDir = os.listdir('result/detection')
        result_paths = []
        for result_path in result_pathDir:
            result_paths.append('./result/detection/'+result_path)
        predict_scaphoid_img_path = result_paths
        self.ui.horizontalSlider.setMaximum(len(pathDir)-1)
        pos= self.ui.horizontalSlider.value()
        #self.ui.slider_label.setText(str(pos+1)+' / '+str(len(pathDir)))
        
        # show original image
        img = cv2.imread(scaphoid_img_path[pos])
        image_height, image_width, image_depth = img.shape
        Qimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Qimg = QtGui.QImage(Qimg.data, image_width, image_height, image_width * image_depth,QtGui.QImage.Format_RGB888)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(Qimg))
        
        # show detect scaphoid_slice
        img_pre_scaphoid = cv2.imread(predict_scaphoid_img_path[pos])
        image_height, image_width, image_depth = img_pre_scaphoid.shape
        Qimg = cv2.cvtColor(img_pre_scaphoid, cv2.COLOR_BGR2RGB)
        Qimg = QtGui.QImage(Qimg.data, image_width, image_height, image_width * image_depth,QtGui.QImage.Format_RGB888)
        self.ui.label_2.setPixmap(QtGui.QPixmap.fromImage(Qimg))
    def pre(self):
        #os.system('python image_id.py')
        #os.system('python json2xml.py')
        #os.system('python tools/dataset_converter/demo_annotation.py --dataset_path=Demo  --classes_path=configs/classes.txt')
        
        #os.system('python eval.py --model_path=model.h5 --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/classes.txt --model_image_size=416x416 --eval_type=VOC --iou_threshold=0.4 --conf_threshold=0.2 --annotation_file=2007_test.txt --save_result')
        #os.system('python yolo_detect.py --model_path=model.h5 --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/classes.txt --model_image_size=416x416 --eval_type=VOC --iou_threshold=0.4 --conf_threshold=0.2 --annotation_file=Demo/2007_demo.txt --save_result')
        
        os.system('python demo_data.py')
        
        output = subprocess.check_output('python yolo_detect.py --model_path=model.h5 --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/classes.txt --model_image_size=416x416 --eval_type=VOC --iou_threshold=0.4 --conf_threshold=0.2 --annotation_file=2007_demo.txt --save_result', shell = True)
        print('output---------------------------------------')
        print(output)
        output = str(output)
        output_test = output.split('Pascal VOC')[-1]
        print('outputtest 0000000000000000000000000000000000000000000---------------------------------------')
        print(output_test)
        print('outputtest wwwwwwwwwwwwwwwwwwwwwwwwwww00---------------------------------------')
        print(output_test.split('\\r\\n'))
        str_out = output_test.split('\\r\\n')
        print(str_out[0])
        print(str_out[1])
        print(str_out[2])
        print(str_out[3])
        print(str_out[4])
        print(str_out[5])
        #self.ui.textBrowser.setText(" "+output_test)
        self.ui.label_3.setText(str_out[0])
        self.ui.label_5.setText(str_out[1])
        self.ui.label_6.setText(str_out[2])
        self.ui.label_7.setText(str_out[3])
        self.ui.label_8.setText(str_out[4])
        self.ui.label_9.setText(str_out[5])
        #print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
        #print(output_test.replace('\r\n', '-'))
        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    