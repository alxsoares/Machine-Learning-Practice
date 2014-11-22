from numpy import *
inputs = array([[0,0], [0,1], [1,0], [1,1]])
targets = array([[0], [1], [1], [1]])
import pcn
p = pcn.pcn(inputs, targets)
p.pcntrain(inputs, targets, 0.25, 6)