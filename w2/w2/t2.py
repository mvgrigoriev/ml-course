# -*- coding: utf-8 -*-
"""
Created on Sun Jan 07 ‏‎11:57:24 2018

@author: MVGrigoriev

@task: Metric select
"""

# At what p the quality on cross-validation turned out to be optimal?  
#
import numpy as np
from sklearn.neighbors import KNeighborsRegressor   # Import class from scikit-learn
from sklearn.datasets import load_boston   # Import boston house-prices dataset
from sklearn.model_selection import cross_val_score   # Import metrics for cross validation
from sklearn.preprocessing import scale   # Import Scale function
from sklearn.model_selection import KFold   # Import KFold function  
boston = load_boston()
scaledData = scale(boston.data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pRange = np.linspace(1, 10, num=200)
listOfAccuracy = []
for i in pRange:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=i)
    #neigh.fit(scaledData, boston.target)
    cvs = cross_val_score(neigh, scaledData, boston.target, cv=kf, scoring='neg_mean_squared_error')
    cvsValue = np.max(cvs)
    listOfAccuracy.append(cvsValue)
optValue = max(listOfAccuracy)
optIndex = listOfAccuracy.index(optValue)
with open('2_5.txt', 'w') as f:
    print(round(pRange[optIndex], 2), file=f, end='')