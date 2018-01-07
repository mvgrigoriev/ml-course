# -*- coding: utf-8 -*-
"""
Created on Sun Jan 07 07:58:16 2018

@author: MVGrigoriev

@task: kNN method
"""

import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier   # Import class from scikit-learn
from sklearn.model_selection import KFold   # Import KFold function  
from sklearn.model_selection import cross_val_score   # Import metrics for cross validation
from sklearn.preprocessing import scale   # Import Scale function  
data = pandas.read_csv('wine.data', header=None)  # Import data
target = data[0]   # Extract target  
features = data.drop(0, axis=1)   # Extract features  
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# At what k is the maximum quality obtained without normalization of characteristics?  
#   
# What is the maximum quality without the normalization of characteristics (the number in the scale from 0 to 1)?  
#   
listOfAccuracy = []
for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(features, target)
    cvs = cross_val_score(neigh, features, target, cv=kf, scoring='accuracy')
    cvsValue = np.mean(cvs)
    listOfAccuracy.append(cvsValue)
optValue = max(listOfAccuracy)
optIndex = listOfAccuracy.index(optValue)
with open('2_1.txt', 'w') as f1:
    print(optIndex+1, file=f1, end='')
with open('2_2.txt', 'w') as f2:
    print(round(optValue, 2), file=f2, end='')

# Which optimal K is obtained after the normalization of the characteristics?  
#   
# What is the maximum quality after the normalization of characteristics (a number in the range from 0 to 1)?  
#   
features = scale(features)
listOfAccuracy = []
for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(features, target)
    cvs = cross_val_score(neigh, features, target, cv=kf, scoring='accuracy')
    cvsValue = np.mean(cvs)
    listOfAccuracy.append(cvsValue)
optValue = max(listOfAccuracy)
optIndex = listOfAccuracy.index(optValue)
with open('2_3.txt', 'w') as f3:
    print(optIndex+1, file=f3, end='')
with open('2_4.txt', 'w') as f4:
    print(round(optValue, 2), file=f4, end='')
