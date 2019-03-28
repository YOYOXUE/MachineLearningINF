
from numpy import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def loadData():
    train_x = []
    train_y = []
    fileIn = open('linear-regression.txt')
    for line in fileIn.readlines():
        lineArr = line.split(',')
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()

dataSet_x, dataSet_y = loadData()
regr=LinearRegression().fit(dataSet_x, dataSet_y)
print(regr.coef_)
