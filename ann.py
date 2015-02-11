import math
import numpy as np
import random
#neural network

def randomWeight():
    return random.random()

def transferFunction(sum):
    return math.tanh(sum)

def transferFunctionDerivative(sum):
    return 1-sum**2

class Connection(object):
    def __init__(self):
        self.weight = 0.0
        self.deltaWeight = 0.0


class Neuron(object):
    def __init__(self, numOutputs, index):
        self._outputVal = None
        self._outputWeights = []
        self._index = index
        self._gradient = 0.0
        self._eta = 0.15 # [0.0 -> 1.0]
        self._alpha = 0.5 # [0.0 -> h] momentum
        #randomly intialize outut weights
        for c in xrange(0, numOutputs):
            self._outputWeights.append(Connection())
            self._outputWeights[-1].weight = randomWeight()


    def _sumDOW(self, nextLayer):
        sum = 0.0
        #sum contribution
        for neuronNum in xrange(0, len(nextLayer)-1):
            sum = sum + self._outputWeights[neuronNum].weight * nextLayer[neuronNum]._gradient
        return sum


    def setOutputVal(self, val):
        self._outputVal = val

    def getOutputVal(self):
        return self._outputVal

    def feedForward(self, prevLayer):
        '''
        Take weight*value for each neruon and connection and sum them up
        :param prevLayer: list of neurons from the previous layer
        '''
        sum = 0
        for neuronNum in xrange(0, len(prevLayer)):
            '''weight_to_neuron is the weight from a neuron at the previous layer
            to this current neuron that's stored in the current Neuron class object'''
            weight_to_neuron = prevLayer[neuronNum]._outputWeights[self._index].weight

            #suming W*I
            sum = sum + prevLayer[neuronNum].getOutputVal() * weight_to_neuron

        self._outputVal = transferFunction(sum)


    def calcOutputGradients(self, targetVal):
        delta = targetVal - self._outputVal
        self._gradient = delta * transferFunctionDerivative(self._outputVal)

    def calcHiddenGradients(self, nextLayer):
        #derivative of weights
        dow = self._sumDOW(nextLayer)
        self._gradient = dow * transferFunctionDerivative(self._outputVal)


    def updateInputWeights(self, prevLayer):

        for neuronNum in xrange(0, len(prevLayer)-1):
            neuron = prevLayer[neuronNum]
            oldDeltaWeight = neuron._outputWeights[self._index].deltaWeight

            newDeltaWeight = \
                self._eta \
                * neuron.getOutputVal() \
                * self._gradient \
                + self._alpha \
                * oldDeltaWeight

            neuron._outputWeights[self._index].deltaWeight = newDeltaWeight
            neuron._outputWeights[self._index].weight = neuron._outputWeights[self._index].weight + newDeltaWeight


class Net(object):
    def __init__(self, topology):
        self._layers = [] # list of lists
        self._error = 0.0
        self._recentAvgError = 0.0
        self._recentAvgSmoothingFactor = 0.0

        #insert layers
        for i in xrange(0, len(topology)):
            layer = topology[i]
            self._layers.append([])

            if i == len(topology)-1:
                numOutputs = 0
            else:
                numOutputs = topology[i+1]

            #inser a neruon for each layer
            for neuronNum in xrange(0, layer+1):
                self._layers[-1].append(Neuron(numOutputs, neuronNum))

            #set bias neuron's output to 1, (last neuron to be pushed)
            self._layers[-1][-1].setOutputVal(1.0)

    def feedForward(self, inputVals):
        '''
        Loops through all the neurons and makes each one feed for ward,
        one layer at a time
        :param inputVals: input layer values (excluding weight)
        '''
        #check size-1 because theres an extra bias neruon!
        assert(len(inputVals) == len(self._layers[0])-1)

        #assign input values to input neruons (layer 0)
        for neuronNum in xrange(0, len(inputVals)):
            self._layers[0][neuronNum].setOutputVal(inputVals[neuronNum])

        #now feed forward from first hidden layer (layer 1)
        for layerNum in xrange(1, len(self._layers)):
            prevLayer = self._layers[layerNum-1]

            #we loop 0 to N-1 neurons, because we don't want to feed forward the bias neuron
            for neuronNum in xrange(0, len(self._layers[layerNum])-1):
                self._layers[layerNum][neuronNum].feedForward(prevLayer)

    def backProp(self, targetVals):
        """
        Back propagation algorithm
        :param targetVals:
        :return:
        """

        #first calculate the overall net error at the output
        outputLayer = self._layers[-1]
        self._error = 0.0

        #calculate RMS error
        for neuronNum in xrange(0, len(targetVals)):
            delta = targetVals[neuronNum] - outputLayer[neuronNum].getOutputVal()
            self._error = self._error + float(delta)**2
        self._error = self._error/(len(outputLayer) - 1)
        self._error = math.sqrt(self._error)

        #metric for recent average measurement
        self._recentAvgError = \
            (self._recentAvgError * self._recentAvgSmoothingFactor + self._error) \
            / (self._recentAvgSmoothingFactor + 1.0)

        #calcualte the output gradients by looping through all neurons
        for neuronNum in xrange(0, len(outputLayer)-1): # -1 because we don't want to loop the bias neuron
            #pass in the target value of the output neuron
            outputLayer[neuronNum].calcOutputGradients(targetVals[neuronNum])

        #calculate gradients for hidden layers
        for layerNum in xrange(len(self._layers)-2, 0, -1): # -2 because we start from the first HIDDEN layer and loop forward toward the input layer
            hiddenLayer = self._layers[layerNum]
            nextLayer = self._layers[layerNum+1]

            for neuronNum in xrange(0, len(hiddenLayer)):
                hiddenLayer[neuronNum].calcHiddenGradients(nextLayer)

        #update connection weights for ALL weights up to the input
        for layerNum in xrange(len(self._layers) - 1, 0, -1):
            layer = self._layers[layerNum]
            prevLayer = self._layers[layerNum-1]

            for neuronNum in xrange(0, len(layer)-1):
                layer[neuronNum].updateInputWeights(prevLayer)

    def getResults(self, results):

        for neuronNum in xrange(0, len(self._layers[-1])-1):
            results.append(self._layers[-1][neuronNum].getOutputVal())

    def getRecentAvgErr(self):
        return self._recentAvgError



def printTestCase(myNet, a):
    result = []
    myNet.feedForward(a)
    myNet.getResults(result)
    print a
    print result[0]

if __name__ == '__main__':
    topology = [2, 2, 1]  #3 in the input layer, 2 in the second, 1 output
    inputVals = []
    targetVals = []
    resultVals = []

    inputs = np.genfromtxt('train.csv', delimiter=',')
    map(np.random.shuffle, inputs)
    myNet = Net(topology)

    print "************************* TRAIN PHASE *************************"
    nPass = 0
    for entry in inputs:
        nPass = nPass+1
        print "***************************************************************"
        print "Pass number " + str(nPass)+":"
        print entry

        #propagate the data
        myNet.feedForward([entry[0], entry[1]])

        #Collect output
        result = list()
        myNet.getResults(result)
        print [entry[0], entry[1], result[0]]

        #train on the data
        target = [entry[2]]
        myNet.backProp(target)

        print "Avg Error: " + str(myNet.getRecentAvgErr())

    print "******************** TEST PHASE ********************"
    a = [1,1]
    b = [1,0]
    c = [0,1]
    d = [1,1]

    printTestCase(myNet, a)
    printTestCase(myNet, b)
    printTestCase(myNet, c)
    printTestCase(myNet, d)


