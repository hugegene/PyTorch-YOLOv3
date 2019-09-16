# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:47:07 2019

@author: bdgecyt
"""

from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import train_test_split

mypath = "C:\\Users\\bdgecyt\\Documents\\GitHub\\PyTorch-YOLOv3\\data\\custom\\images"
onlyfiles = np.array([f for f in listdir(mypath) if f.endswith(".jpg")])

#one = np.array([2,3,4,5])
#print(len(onlyfiles))

x_train ,x_valid = train_test_split(onlyfiles,test_size=0.3)  

print(x_train)
print(x_valid)

f= open("C:\\Users\\bdgecyt\\Documents\\GitHub\\PyTorch-YOLOv3\\data\\custom\\train.txt","w+")
for i in x_train:
     f.write("data/custom/images/"+i+"\n")
f.close() 

f= open("C:\\Users\\bdgecyt\\Documents\\GitHub\\PyTorch-YOLOv3\\data\\custom\\valid.txt","w+")
for i in x_valid:
     f.write("data/custom/images/"+i +"\n")
f.close() 

