# mlp for regression
import numpy as np
import pylab as pl

x = np.linspace(0,1,40).reshape((40,1))
t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40).reshape((40,1))*0.2
x = (x-0.5)*2

# pl.plot(x, t, '')
# pl.show()

train = x[0::2, :]
test = x[1::4, :]
valid = x[3::4, :]
traintarget = t[0::2, :]
testtarget = x[1::4, :]
validtarget = x[3::4, :]

pl.plot(x, t, 'o')
pl.xlabel('x')
pl.ylabel('t')

import mlp
net = mlp.mlp(train, traintarget, 3, outtype='linear')
net.mlptrain(train, traintarget, 0.25, 101)

net.earlystopping(train,traintarget,valid,validtarget,0.25)

pl.show()