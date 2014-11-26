import numpy as np
import mlp

data1 = np.loadtxt('../Data/dataset1.txt')
data2 = np.loadtxt('../Data/dataset2.txt')
data3 = np.loadtxt('../Data/dataset3.txt')

train = data1[:, 0:2]
traintarget = data1[:, 2]
traintarget.shape = (traintarget.shape[0], 1)

valid = data2[:, 0:2]
validtarget = data2[:, 2]
validtarget.shape = (validtarget.shape[0], 1)

test = data3[:, 0:2]
testtarget = data3[:, 2]
testtarget.shape = (testtarget.shape[0], 1)

net = mlp.mlp(train, traintarget, 3, outtype='logistic')
# net.mlptrain(train, traintarget, 0.1, 200)
net.earlystopping(train, traintarget, valid, validtarget, eta=0.1, niterations=100)
net.confmat(test, testtarget)

