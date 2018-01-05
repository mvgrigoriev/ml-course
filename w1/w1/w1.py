
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 07:34:46 2017

@author: MVGrigoriev

@task: Data preprocessing in Pandas

"""

import pandas
import os
data = pandas.read_csv('titanic.csv', index_col='PassengerId')   # Data import

# How many men and women were traveling by ship?
#
sexCounts = data['Sex'].value_counts()
with open('1_1.txt', 'w') as f1:
    print(sexCounts[0], sexCounts[1], file=f1, end='')

# What part of the passengers managed to survive?
#
survivePassengers = data['Survived'].value_counts()
with open('1_2.txt', 'w') as f2:
    print(round(100. * survivePassengers[1] / len(data), 2), file=f2, end='')   # Round values

# What percentage of the first class passengers were among all passengers?
#
firstClassPassengers = data['Pclass'].value_counts()
with open('1_3.txt', 'w') as f3:
    print(round(100. * firstClassPassengers[1] / len(data), 2), file=f3, end='')

# How old were the passengers?
#
with open('1_4.txt', 'w') as f4:
    print(round(data['Age'].mean(), 2), data['Age'].median(), file=f4, end='')   # Average and median

# Do the number of brothers / sisters / spouses correlate with the number of
# parents / children?
#
with open('1_5.txt', 'w') as f5:
    print(round(data['SibSp'].corr(data['Parch']), 2), file=f5, end='')

# What is the most popular female name on the ship?
#
females = data.where(data.Sex == 'female')
femaleNames = females['Name']
nameOnly = femaleNames.str.split('\. ', expand=True)
allNames = nameOnly[1].value_counts().index.tolist()   # Out names to list
with open('1_6.txt', 'w') as f6:
    print(allNames[0], file=f6, end='')