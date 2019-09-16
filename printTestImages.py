# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:41:39 2019

@author: bdgecyt
"""
from shutil import copyfile

# Open the file with read only permit
f = open('data\\custom\\valid.txt', "r")


line = f.readline()
# use the read line to read further.
# If the file is not empty keep reading one line
# at a time, till the file is empty
count = 0
while line:
    
    source = line
    print(line[-12:])
    dst = "data\\custom\\" +line[-12:]
    if count%10 == 0:
        copyfile(source, dst)
        

    count +=1
    line = f.readline()
    
    
    
f.close()

