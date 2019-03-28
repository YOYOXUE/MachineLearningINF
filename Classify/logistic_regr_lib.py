


from numpy import *
import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression

def loadData():
    train_x = []
    train_y = []
    fileIn = open('classification.txt')
    for line in fileIn.readlines():
        lineArr = line.split(',')
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1]),float(lineArr[2])])
        tmp=int(lineArr[4])
        if tmp==-1:
            train_y.append(0)
        else:
            train_y.append(tmp)
    train_y=np.array(train_y)
    #print(type(train_y))
    return mat(train_x), train_y#mat(train_y).transpose()

x,y= loadData()
logreg=LogisticRegression()
logreg.fit(x,y)
print(logreg.coef_)
print(logreg.score(x,y))
