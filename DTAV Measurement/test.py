# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:43:45 2022

@author: ZLS
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:14:42 2022

@author: ZLS
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 21:51:28 2022

@author: ZLS
"""
# -*- coding: utf-8 -*-
import os.path
import glob
import cv2
import numpy as np
import math

if __name__ == "__main__":
    seg_y = cv2.imread('./1.png')      # segmentation results  
    canny_rectum = cv2.imread('./canny.png')    # segmentation rectum results after canny    
    height = seg_y.shape[0]
    weight = seg_y.shape[1]   
    x_tumor =0
    y_tumor =0
    bot =[]
    begin_x =0
    for row in range(height):            
       for col in range(weight):        
            if seg_y[row, col, 0] == 0  and seg_y[row, col, 2] == 0 and seg_y[row, col, 1] == 255:
                x_tumor = col
                y_tumor = row     

    for row in range(y_tumor,height):            
       for col in range(weight):     
            if canny_rectum[row, col, 0]==255 or  canny_rectum[row, col, 1]==255 or canny_rectum[row, col, 2]==255:
                   seg_y[row, col, 0] = 255
                   seg_y[row, col, 2] = 255
                   seg_y[row, col, 1] = 255
                   break 
    cv2.imwrite(save_path+str(image_name),seg_y)
    #c_x:nx,c_y:ny,c_xy:nxy
    z = 0
    c_x = 0
    c_y = 0
    c_xy = 0
    #print(height)
    #print(weight)
    for row in range(height):           
           for col in range(weight):        
                if seg_y[row, col, 0] >= 200  and seg_y[row, col, 2] >=200 and seg_y[row, col, 1] >= 200:
                    z+=1
    kepoint=[]
    for row in range(height):            
           for col in range(weight):         
                if seg_y[row, col, 0] >= 200  and seg_y[row, col, 2] >= 200 and seg_y[row, col, 1] >= 200:
                   kepoint.append ((col,row))
    
    for i in range(len(kepoint)-1):
        j= i+1
        if kepoint[i][0] == kepoint[j][0] and kepoint[i][1] != kepoint[j][1]:
           c_x+=1
        if kepoint[i][0] != kepoint[j][0] and kepoint[i][1] == kepoint[j][1]:
           c_x+=1
        if kepoint[i][0] != kepoint[j][0] and kepoint[i][1] != kepoint[j][1]:
           c_xy+=1
    id_ = image_name.split('IMG')[0]
    #save info_nx,ny,nxy
    save_path_info = "./nx_ny_nxy.txt"   
    fname = open(save_path_info,'w')
    fname.write(str(id_) + ' '+str(z)+ ' '+ str(c_x)+ ' '+ str(c_y)+ ' '+ str(c_xy)+ ' '+str(height) + '\n') 
    
    f=open("./info_dxy.txt")#info_dx,dy,dxy
    
    f1=open("./nx_ny_nxy.txt")#info_nx,ny,nxy
    save_path_res = "./res_DTAV.txt"#DTAV_res
    fname1 = open(save_path_res,'w')
    data = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
    data1 = f1.readlines()
    patient_dx =[]
    patient_dy =[]
    patient_dxy =[]
    for i in range(len(data)):
        print(data[i].split('\t'))
        if len(data[i].split('\t')[1]) !=0:
            print('data[i].split(' ')[0]',data[i].split('\t')[0])
            patient_dx.append((data[i].split('\t')[0],data[i].split('\t')[1]))
            print('data[i].split(' ')[1]',data[i].split('\t')[1])
            patient_dy.append((data[i].split('\t')[0],data[i].split('\t')[2]))
            print('data[i].split(' ')[2]',data[i].split('\t')[2])
            patient_dxy.append((data[i].split('\t')[0],data[i].split('\t')[3]))
            print('data[i].split(' ')[3]',data[i].split('\t')[3])
    f.close()  # 关
    #返回list
    num_patient_dx =[]
    num_patient_dy =[]
    num_patient_dxy =[]
    print('data1',data1)
    for i in range(len(data1)):
        if len(data1[i].split(' ')[1]) !=0:
            #print('data1[i].split(' ')[0]',data1[i].split(' ')[0])
            num_patient_dx.append((data1[i].split(' ')[0],data1[i].split(' ')[2]))
            #print('data1[i].split(' ')[2]',data1[i].split(' ')[2])
            num_patient_dy.append((data1[i].split(' ')[0],data1[i].split(' ')[3]))
            #print('data1[i].split(' ')[3]',data1[i].split(' ')[3])
            num_patient_dxy.append((data1[i].split(' ')[0],data1[i].split(' ')[4]))
            #print('data1[i].split(' ')[4]',data1[i].split(' ')[4])
    
    res =[]
    print('num_patient_dx',num_patient_dx)
    print('patient_dx',patient_dx)
    
    for i in range(len(num_patient_dx)):
        for j in range(len(patient_dx)):
            #print('patient_dx[j][0]',patient_dx[j][0])
            #print('num_patient_dx',num_patient_dx[i][0])
            if num_patient_dx[i][0] == patient_dx[j][0]:
               print('x',num_patient_dx[i][0])
               res.append((num_patient_dx[i][0],(float(num_patient_dx[i][1])*float(patient_dx[j][1])+(float(num_patient_dxy[i][1])*float(patient_dxy[j][1])))/10))
    print('res',res)
    for i in range(len(res)):
        fname1.write(res[i][0] + ' '+ 'DTAV：' + str(res[i][1]) + ' cm' + '\n')
        # save info id and DTAV value in res_DTAV.txt
