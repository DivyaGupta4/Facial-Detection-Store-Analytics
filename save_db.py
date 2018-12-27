# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:06:00 2018

@author: Divya
"""

import time
import pandas as pd
import datetime 

gender_list = []
emotion_list = []
faces_list = []
with open('webcam.log') as log:
    lines = log.readlines()
    
    for line in lines:
        a = line[10:-1]
        if a in ['M','F']:
            gender_list.append(a)
        elif a in ['Happy','Sad','Neutral','Surprised']:
            emotion_list.append(a)
        else:
            faces_list.append(a)
    
current_time = str(datetime.datetime.now().date())

a = pd.Series(gender_list).value_counts().sort_values(ascending=False).index[0]

b = pd.Series(emotion_list).value_counts().sort_values(ascending=False).index[0]

c = pd.Series(faces_list).value_counts().sort_values(ascending=False).index[0]

wr = current_time+","+a +","+b+","+c

with open('upload.csv','a') as file:
    file.write('\n'+wr)