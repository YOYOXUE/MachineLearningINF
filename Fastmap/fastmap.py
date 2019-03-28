import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
#Yidan_Xue&Wenjie_Shi
#Author:Yidan_Xue
class FastMap:

    def __init__(self, targe_dimension, newDimension = []):
        self.targe_dimension = targe_dimension
        self.newDimension = newDimension

    def train(self,data):
        dist_array = np.zeros((10, 10), dtype = np.int)

        for item in data:
            i = item[0]
            j = item[1]
            dist_array[i-1][j-1] = item[2]
            dist_array[j-1][i-1] = item[2]
#         print(dist_array)
        self.fastMap(self.targe_dimension, dist_array)

    def farthest_point(self,distance):

        o1 = np.random.randint(0,9)
        while True:
            farthest = max(distance[o1])
            o2 = distance[o1].index(farthest)
            tmpDistance = max(distance[o2])
            tmpO = distance[o2].index(tmpDistance)
            if (tmpO == o1):
                break
            else:
                o1 = o2

        if o1 < o2:
            return (o1, o2)
        else:
            return (o2, o1)


    def fastMap(self, n, distance):
        if n<=0:
            return
        distance = distance.tolist()
        pivots = self.farthest_point(distance)

        a = pivots[0]
        b = pivots[1]
        farthest = distance[a][b]
        dimension = []

        for i in range(10):
            tmpDistance = 0
            if i == a:
                dimension.append(0)
            elif i == b:
                dimension.append(farthest)
            else:
                tmpDistance = ((distance[a][i]**2) + (farthest**2) - (distance[b][i]**2))/float(2 * farthest)
                dimension.append(tmpDistance)

        self.newDimension.append(dimension)
        projection = np.zeros((10, 10))

        if (n >= 1):
            for i in range (10):
                for j in range (10):
                    tmp = (distance[i][j] ** 2) - ((dimension[i] - dimension[j]) ** 2)
                    projection[i][j] = np.sqrt(tmp)
            self.fastMap(n-1, projection)

def main():
    readfile = "fastmap-data.txt"
    nameFile = "fastmap-wordlist.txt"
    with open('fastmap-data.txt','r') as f:
        lines=[]
        alllines=f.readlines()
        for i in alllines:
            line=i.rstrip().split('\t')
            l=[]
            for i in line:
                a=int(i)
                l.append(a)
            lines.append(l)
#         print(lines)
        data=np.array(lines)
    with open("fastmap-wordlist.txt",'r') as fd:
        name = fd.read()
        nameList = name.strip().split('\n')
#     data = pd.read_csv(readfile, sep = '\t', header = None)

    target_dimension = 2

    fm = FastMap(target_dimension)
    fm.train(data)
    newDimension1 = np.array(fm.newDimension)
    newDimension1=newDimension1.T

    print(newDimension1)


    fig,ax = plt.subplots()

    ax.scatter(newDimension1[:,0], newDimension1[:,1])
    i=0
    for txt in enumerate(nameList):
        ax.annotate(txt[1], (newDimension1[i][0], newDimension1[i][1]))
        i+=1
    plt.show()

if __name__ == '__main__':
    main()
