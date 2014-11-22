import numpy as np
import linreg

auto = np.loadtxt('../Data/auto-mpg.data', comments='"')

# Normalise the data
auto[:,0:-1] *= 1./auto[:,0:-1].max(axis=0)

# Separate the data into training and testing sets

trainin = auto[0:round(np.shape(auto)[0]*0.4),:]
traintgt = trainin[:,-1].reshape(-1,1)
trainin = trainin[:,0:-1]

testin = auto[round(np.shape(auto)[0]*0.4):,:]
testtgt = testin[:,-1].reshape(-1,1)
testin = testin[:,0:-1]


# This is the training part
beta = linreg.linreg(trainin,traintgt)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
testout = np.dot(testin,beta)
error = np.sum((testout - testtgt)**2)
print error
