import numpy as np
import mlp

data1 = np.loadtxt('../Data/dataset1.txt')
data2 = np.loadtxt('../Data/dataset2.txt')
data3 = np.loadtxt('../Data/dataset3.txt')

train = data1[:, 0:2]
train = np.asmatrix(train)
traintarget = data1[:, 2]
traintarget = np.transpose(np.asmatrix(traintarget))
test = data2[:, 0:2]
test = np.asmatrix(test)
testtarget = data2[:, 2]
testtarget = np.asmatrix(testtarget)

net = mlp.mlp(train, traintarget, 3, outtype='logistic')
net.confmat(test, testtarget)
