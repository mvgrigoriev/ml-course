# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 ‏‎19:11:07 2018

@author: MVGrigoriev

@task: Support objects
"""

# Find the numbers of objects that are supports.    
#
import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

trainData = pandas.read_csv('svm-data.csv', header=None) 
targetTrainData = trainData[0]
featuresTrainData = trainData.drop(0, axis=1)

clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(featuresTrainData, targetTrainData)

listOfIndexes = []
for i in clf.support_:
    listOfIndexes.append(str(i+1))

with open('3_1.txt', 'w') as f:
    print(','.join(listOfIndexes), file=f, end='')
