
import numpy as np
from numpy import *
import pandas as pd

def loadData():
    train_x = []
    train_y = []
    fileIn = open('linear-regression.txt')
    for line in fileIn.readlines():
        lineArr = line.split(',')
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()

x, y = loadData()
m,n = np.shape(x)
lr = 0.07
theta = np.mat(np.zeros(n))
for i in range(20000):
    h = np.dot(x,theta.transpose())
    det_h = h-y
    det_t = lr*(1.0/m)*np.dot(det_h.transpose(),x)
    theta = theta-det_t
print("weight:")
print(theta)


total=np.dot(x,theta.transpose())
loss=total-y
l=0
for i in loss:
    l+=i*i
l=1-l/len(y)
print("accuracy:")
print(l)
