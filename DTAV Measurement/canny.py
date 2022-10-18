# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 21:34:10 2022

@author: ZLS
"""

import os.path
import glob
import cv2
import numpy as np

source_path = "G:/MRI_DTAV/patterns/code/DTAV/up/seg/"       
save_path = "G:/MRI_DTAV/patterns/code/DTAV/up/seg/1/"    
M =0
for item in glob.glob(source_path+"*.png"):  
    image_name = os.path.split(item)[1]         # os.path.split按照路径将文件名和路径分割开
    #print('image_name',image_name)
    #print('item',item)
    image = cv2.imread(item,0)                 # image是numpy.ndarray ; 以灰度模式加载图片
    #image = cv2.imread(item)
    #cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=15), cv2.COLORMAP_HSV)
    #print('image',image)
    image_ = cv2.Canny(image, 50, 150,L2gradient=True) 
    cv2.imwrite(save_path+str(image_name),image_)   