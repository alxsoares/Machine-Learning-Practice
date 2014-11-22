import numpy as np

class pcn:
    """
    A basic Perceptron
    """

    def __init__(self, inputs, targets):
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else:
            self.nIn = 1

        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]

        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)

        for n in range(nIterations):

            self.activations = self.pcnfwd(inputs)
            self.weights -= eta * np.dot(np.transpose(inputs), self.activations - targets)
            print "Iteration: ", n
            print self.weights

        activations = self.pcnfwd(inputs)
        print "Final outputs are:"
        print activations


    def pcnfwd(self, inputs):
        """ Run the network forward """
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)

    def confmat(self, inputs, targets):
        """ Confusion matrix """
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        outputs = np.dot(inputs, self.weights)

        nClasses = np.shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print cm
        print np.trace(cm) / np.sum(cm)