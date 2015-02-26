'''
    ann.py - Neural network class that is responsible for forward propagation and training (back propagatin)

'''

from neuron import *
from jANN_utils import *

import numpy as np
from sklearn.utils import shuffle


class Net(object):
    '''
    Basic Neural Network class. Always adds a bias neuron to each level. The last level doesn't use the bias neuron.
    This network is created as a fully connected network
    '''
    def __init__(self, topology):
        '''
        Constructor
        :param topology: LIST of neuron counts for each layer in the order
        [input_layer, hidden, hiddner, ... , output_layer]
        '''
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
        :param targetVals: LIST of target values.
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

    def train(self, inputVals, target):
        '''
        Train towards target given inputVals
        :param inputVals: LIST of input values
        :param target:  LIST of target values
        '''
        #propagate the input values forward in the net
        self.feedForward(inputVals)

        #train on the data by correcting for an amount of error
        self.backProp(target)

    def test(self, inputVals):
        '''
        Test input values to see what they return from the network
        :param inputVals: LIST of input values
        :return: LIST of output values
        '''
        outputVals = []
        self.feedForward(inputVals)
        self.getResults(outputVals)
        return outputVals

    def getResults(self, results):
        '''
        Gets the outputs for all the neruons at the output layer
        TODO: Use return to output results instead of this passing thing.
        :param results:
        :return:
        '''
        for neuronNum in xrange(0, len(self._layers[-1])-1):
            results.append(self._layers[-1][neuronNum].getOutputVal())

    def getRecentAvgErr(self):
        return self._recentAvgError




if __name__ == '__main__':
    topology = [2, 2, 1]  #2 in the input layer, 2 in the second, 1 output

    #Load data and shuffle for randomness
    inputs = np.genfromtxt('train.csv', delimiter=',')
    inputs = shuffle(inputs)
    myNet = Net(topology)

    print "************************* TRAIN PHASE *************************"
    nPass = 0
    for entry in inputs:
        nPass = nPass+1
        print "***************************************************************"
        print "Pass number " + str(nPass)+":"
        print entry

        #train
        inputVals = [float(entry[0]), float(entry[1])]
        target = [entry[2]]
        myNet.train(inputVals, target)

        #Collect output
        result = list()
        myNet.getResults(result)
        print [entry[0], entry[1], result[0]]

        print "Avg Error: " + str(myNet.getRecentAvgErr())

    print "******************** TEST PHASE ********************"
    a = [1.,1.]
    b = [1.,0.]
    c = [0.,1.]
    d = [0.,0.]

    printTestCase(myNet, a)
    printTestCase(myNet, b)
    printTestCase(myNet, c)
    printTestCase(myNet, d)

    #print the resulting network and weights
    drawEdges(myNet)
