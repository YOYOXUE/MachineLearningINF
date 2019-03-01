
from matplotlib import pylab as plt
import numpy as np


def read_file(filename):
    with open(filename,'r') as f:
    #f = open(filename, 'r')
        d = f.readlines()
    return d


def get_circle(centroid, ccov):
    sdwidth = 1
    points = 50
    mean = np.c_[centroid]  # mu
    tt = np.c_[np.linspace(0, 2 * np.pi, points)]  # generate points
    x = np.cos(tt)
    y = np.sin(tt)
    ap = np.concatenate((x, y), axis=1).T
    d, v = np.linalg.eig(ccov)  # variance
    d = np.diag(d)
    d = sdwidth * np.sqrt(d)  # convert variance to sdwidth*sd
    bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1]))
    return bp[0, :], bp[1, :]


def gaussian(x_i, mu_i, sigma_i):
    """
    Multi Gaussian distribution
    """
    norm_factor = 1.0 / np.linalg.det(sigma_i)# Matrix Determinant
    sgm = norm_factor * np.exp(-0.5 * np.transpose(x_i - mu_i).dot(np.linalg.inv(sigma_i)).dot(x_i - mu_i))
    return sgm


class GMM(object):
    """
    Gaussian Mixture Model
    """
    def __init__(self):
        # read data
        data = read_file('clusters.txt')
        self.n = len(data)  # num of total data
        self.dim = 2  # dimension of x
        self.x = np.zeros((self.n, self.dim), dtype='float64')  # data x
#         self.y = np.zeros((self.n, 1), dtype='float64')  # 类别y，行向量是one-hot vector

        for i in range(len(data)):
            data_ = data[i].split(',')
#             self.y[i] = int(data_[0])
            self.x[i][0] = float(data_[0])
            self.x[i][1] = float(data_[1])

        # initialize parameters
        self.k = 3  # num of Gaussian Models
        self.gama = np.zeros((self.n, self.k), dtype='float64')
        self.mu = np.zeros((self.k, self.dim), dtype='float64')  # mean
        self.mu[0] = self.x[49]
        self.mu[1] = self.x[149]
        self.mu[2] = self.x[0]

        self.sigma = np.array([[[1, 0], [0, 1]] for i in range(0, self.k)], dtype='float64')  # 协方差
        self.alpha = np.array([1.0 / self.k for i in range(0, self.k)], dtype='float64')  # 混合比
        self.epsilon = 0.000001

        self.em()
        self.gmm()

    def new_mu(self, k):
        """
        calculate new mean：mu_k
        xi：data[i]
        """
        mu_k = np.zeros((1, self.dim), dtype='float64')
        for i in range(self.n):
            mu_k += self.gama[i][k] * self.x[i]
        return mu_k / np.sum(self.gama[:, k])

    def new_sigma(self, k):
        """
        calculate new sigma
        """
        sigma_k = np.zeros((self.dim, self.dim), dtype='float64')
        for i in range(self.n):
            sigma_k += self.gama[i][k] * np.transpose(np.mat(self.x[i] - self.mu[k])).\
                dot(np.mat(self.x[i] - self.mu[k]))
        return sigma_k / np.sum(self.gama[:, k])

    def new_alpha(self, k):
        """
        calculate new alpha:mixture proportion计算新的混合比
        """
        return np.sum(self.gama[:, k]) / self.n

    def Q(self):
        """
        计算优化目标的损失函数
        """
        q = 1
        for i in range(self.n):
            p = 0
            for j in range(self.k):
                p += self.alpha[j] * gaussian(self.x[i], self.mu[j], self.sigma[j])
            q *= p
        return q

    def em(self):
        """
        Expectation Maximization Algorithm
        """
        p_new = self.epsilon * 2
        p_old = 0
        while abs(p_new - p_old) > self.epsilon:
            p_old = p_new
            for k in range(20):
                # E Step
                for i in range(self.n):
                    sum_gama_k = 0
                    for j in range(self.k):
                        sum_gama_k += self.alpha[j] * gaussian(self.x[i], self.mu[j], self.sigma[j])

                    for j in range(self.k):
                        #r(i,k)
                        self.gama[i][j] = (self.alpha[j] * gaussian(self.x[i], self.mu[j], self.sigma[j])) / sum_gama_k

                # M Step
                for i in range(self.k):
                    self.alpha[i] = self.new_alpha(i)
                    self.sigma[i] = self.new_sigma(i)
                    self.mu[i] = self.new_mu(i)
                p_new = self.Q()

    def gmm(self):
        """
        根据em中得到的参数进行聚类
        """
        #count how many points in each cluster
        count1 = 0
        count2 = 0
        count3 = 0
        for i in range(self.n):
            p1 = self.alpha[0] * gaussian(self.x[i], self.mu[0], self.sigma[0])
            p2 = self.alpha[1] * gaussian(self.x[i], self.mu[1], self.sigma[1])
            p3 = self.alpha[2] * gaussian(self.x[i], self.mu[2], self.sigma[2])
            if max(p1, p2, p3) == p1:
                count1 += 1
            elif max(p1, p2, p3) == p2:
                count2 += 1
            else:
                count3 += 1
        #initializa three clusters as 2-dim zero vectors
        x_1 = np.zeros((count1, self.dim), dtype='float64')
        x_2 = np.zeros((count2, self.dim), dtype='float64')
        x_3 = np.zeros((count3, self.dim), dtype='float64')

        #put points belong to a cluster
        count1 = 0
        count2 = 0
        count3 = 0
        for i in range(self.n):
            p1 = self.alpha[0] * gaussian(self.x[i], self.mu[0], self.sigma[0])
            p2 = self.alpha[1] * gaussian(self.x[i], self.mu[1], self.sigma[1])
            p3 = self.alpha[2] * gaussian(self.x[i], self.mu[2], self.sigma[2])
            if max(p1, p2, p3) == p1:
                x_1[count1] = self.x[i]
                count1 += 1
            elif max(p1, p2, p3) == p2:
                x_2[count2] = self.x[i]
                count2 += 1
            else:
                x_3[count3] = self.x[i]
                count3 += 1
        print('mean',self.mu)
        print('conv',self.sigma)
        print('amplitude')
        for i in self.sigma:
            print(1.0 / np.linalg.det(i))
        plt.plot(x_1[:, 0], x_1[:, 1],'bo')
        plt.plot(x_2[:, 0], x_2[:, 1], 'go')
        plt.plot(x_3[:, 0], x_3[:, 1], 'ro')

        # 画曲线
        x1, y1 = get_circle(self.mu[0], self.sigma[0])
        x2, y2 = get_circle(self.mu[1], self.sigma[1])
        x3, y3 = get_circle(self.mu[2], self.sigma[2])

        plt.plot(x1, y1, 'b')
        plt.plot(x2, y2, 'g')
        plt.plot(x3, y3, 'r')

        plt.show()

if __name__ == '__main__':
    a = GMM()
