# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:07:50 2019

@author: bdgecyt
"""

# importing os module 
import os 
  
# Function to rename multiple files 


      
for filename in os.listdir("C:\\Users\\bdgecyt\\Desktop\\hhh"): 
    print(filename)
    src =  filename
    dst = filename[:-2] + "g"
    print(filename)
  
    
    os.rename(src, dst