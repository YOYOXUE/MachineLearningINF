import numpy
import copy
from Perceptron import Perceptron
LEARNINGFACTOR = .5
class PocketPerceptron(Perceptron):
    def __init__(self, indim):
        Perceptron.__init__(self, indim)
        self.pocketweight = numpy.zeros(self.dim + 1)
        self.pocketerror = 1

    def digest(self, allrows, allvals):
        for i in range(len(allvals)):
            row = [1] + allrows[i]
            val = allvals[i]
            prediction = self.calcval(row)
            diff = val - prediction
            if (diff != 0):
                diff = self.stepfunc(diff)

            for j in range(len(self.weights)):
                self.weights[j] = self.weights[j] +  LEARNINGFACTOR * diff * row[j]

        error = self.getError(allrows, allvals)
        if error < self.pocketerror:
            self.pocketerror = error
            self.pocketweight = copy.deepcopy(self.weights)

    def finalize(self):
        self.weights = copy.deepcopy(self.pocketweight)
        # print(self.weights)

    def getError(self, allrows, allvals):
        countwrong = 0
        for i in range(len(allvals)):
            row = allrows[i]
            val = allvals[i]
            prediction = self.predict(row)
            if val != prediction:
                countwrong += 1

        error = float(countwrong) / len(allvals)
        return error

if __name__=="__main__":
    perc = PocketPerceptron(3)
    allrows = []
    allvals = []
    for line in open('classification.txt', 'r'):
        line = line.split(',')
        row = [float(line[0]), float(line[1]), float(line[2])]
        val = int(line[4])
        allrows.append(row)
        allvals.append(val)
    for i in range(7000):
        perc.digest(allrows, allvals)
    perc.finalize()
    print('pocket perceptron weights are:')
    print(perc.weights)
    correct = 0
    total = 0
    for i in range(len(allvals)):
        row = allrows[i]
        val = allvals[i]
        prediction = perc.predict(row)
        total += 1
        if val == prediction:
            correct += 1
    print('pocket perceptron accuracy: ')
    accuracy = float(correct) / total
    print(accuracy)
