# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:17:16 2017

@author: MVGrigoriev

@task: Decision tree
"""

import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier   # Import class from scikit-learn
data = pandas.read_csv('titanic.csv', usecols=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])  # Import data
data = data.dropna()   
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
clf = DecisionTreeClassifier(random_state=241)   # Create classification Decision tree,
                                                 # DecisionTreeRegressor - for regression
features = data[['Pclass', 'Fare', 'Age', 'Sex']]  # Features
target = data[['Survived']] # Target
clf.fit(features, target)   # Learning
featuresName = ['Pclass', 'Fare', 'Age', 'Sex']
importances = clf.feature_importances_
topFeaturesIndeces = sorted(range(len(importances)), key=lambda i: importances[i])[-2:]   # Get indexes of TOP2 values
with open('1_7.txt', 'w') as f:
    print(featuresName[topFeaturesIndeces[0]], featuresName[topFeaturesIndeces[1]], file=f, end='')

