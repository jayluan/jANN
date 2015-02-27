'''
    jANN
    Copyright (c) Jay Luan, All rights reserved.
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 3.0 of the License, or (at your option) any later version.
    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public
    License along with this library.



    Neuron.py - basic neuron class which stores weights connecitons, and outputs
'''

import random
import math

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
        self._alpha = 0.7 # [0.0 -> h] momentum
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

        for neuronNum in xrange(0, len(prevLayer)):
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
