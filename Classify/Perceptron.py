import numpy


class Perceptron:
    def __init__(self, indim):
        self.dim = indim
        self.weights = numpy.zeros(self.dim + 1)

    def digest(self, row, desiredval):
        row = [1] + row
        if len(row) != len(self.weights):
            return []
        expected = self.calcval(row)
        diff = desiredval - expected
        if(diff != 0):

            diff = self.stepfunc(diff)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + rate * diff * row[i]

    def predict(self, row):
        row = [1] + row
        return self.calcval(row)

    def calcval(self, rowplus1):

        val = 0
        for i in range(len(self.weights)):
            val += self.weights[i] * rowplus1[i]
        return self.stepfunc(val)

    def stepfunc(self, val):
        if val > 0:
            return 1
        else:
            return -1

if __name__=="__main__":
    perc=Perceptron(3)
    correct=0
    total=0
    rate=0.01
    with open ('classification.txt','r') as f:
        for line in f:
            line=line.split(',')
            row=[float(line[0]), float(line[1]), float(line[2])]
            val = int(line[3])
            perc.digest(row, val)



    with open ('classification.txt','r') as d:
        for line in d:
            line=line.split(',')
            row=[float(line[0]), float(line[1]), float(line[2])]
            val=int(line[3])
            prediction=perc.predict(row)
            total+=1
            if val==prediction:
                correct+=1
    #         else:
    #             er.append(row)

    print('perception weights are:')
    print(perc.weights)

    print('perception accuracy:')
    accuracy=float(correct)/total
    print(accuracy)
