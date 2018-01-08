# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 16:57:43 2018

@author: MVGrigoriev

@task: Features normalizaion
"""

# Find the difference between the quality on the test sample after the normalization and the quality before it.    
#
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

trainData = pandas.read_csv('perceptron-train.csv', header=None) 
targetTrainData = trainData[0]
featuresTrainData = trainData.drop(0, axis=1)

testData = pandas.read_csv('perceptron-test.csv', header=None)
targetTestData = testData[0]
featuresTestData = testData.drop(0, axis=1)

clf = Perceptron(random_state=241)
clf.fit(featuresTrainData, targetTrainData)

predictions = clf.predict(featuresTestData)
accuracyClassificationScoreBeforeNormalization = accuracy_score(targetTestData, predictions)

scaler = StandardScaler()
featuresTrainDataScaled = scaler.fit_transform(featuresTrainData)
featuresTestDataScaled = scaler.transform(featuresTestData)

clf = Perceptron(random_state=241)
clf.fit(featuresTrainDataScaled, targetTrainData)

predictions = clf.predict(featuresTestDataScaled)
accuracyClassificationScoreAfterNormalization = accuracy_score(targetTestData, predictions)

with open('2_6.txt', 'w') as f:
    print(accuracyClassificationScoreAfterNormalization-accuracyClassificationScoreBeforeNormalization, file=f, end='')
