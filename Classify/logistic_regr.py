

from numpy import *
import pandas as pd

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def trainLogRegres(train_x, train_y):
    numSamples, numFeatures = shape(train_x)
    weights = ones((numFeatures, 1))
    alpha=0.01
    maxIter=7000
    # optimize through gradient descent algorilthm
    for k in range(maxIter):
        output = sigmoid(train_x * weights)
        error = train_y - output
        weights = weights + alpha * train_x.transpose() * error
    return weights

def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


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
    return mat(train_x), mat(train_y).transpose()


x, y = loadData()
weights = trainLogRegres(x, y)
print(weights)
accuracy = testLogRegres(weights, x,y)
print(accuracy)
