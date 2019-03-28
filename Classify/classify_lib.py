# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron

if __name__=="__main__":
    data=[]
    labels=[]
    with open ('classification.txt','r') as f:
        for line in f:
            line=line.split(',')
            row=[float(line[0]), float(line[1]), float(line[2])]
            data.append(row)
            val = int(line[3])
            labels.append(val)
    d=np.array(data)

    net = perceptron.Perceptron(max_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
    net.fit(d,labels)
    # Print the results
    # print("Prediction " + str(net.predict(d)))
    # print("Actual     " + str(labels))
    print("Accuracy   " + str(net.score(d, labels)*100) + "%")
    # print('weights0(intercept)  '+str(net.intercept_))
    print('weights   '+str(list(net.intercept_)+list(net.coef_[0])))

    # Plot the original data
    d90 = np.rot90(d)
    d90 = np.rot90(d90)
    d90 = np.rot90(d90)

    colormap = np.array(['r', 'k'])
    plt.scatter(d90[0], d90[1] ,c=colormap[labels], s=5)

    # Output the values
    # print("Coefficient 0 " + str(net.coef_[0,0]))
    # print("Coefficient 1 " + str(net.coef_[0,1]))
    # print("Bias " + str(net.intercept_))

    # Calc the hyperplane (decision boundary)
    ymin, ymax = plt.ylim()
    w = net.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (net.intercept_[0]) / w[1]

    # Plot the line
    plt.plot(yy,xx, 'k-')
    plt.show()
