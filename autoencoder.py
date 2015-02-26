'''
    autoencoder.py - autoencoder class build on top of the Net class.
'''

from neuron import *
from ann import Net

class AutoEncoder(Net):
    def __init__(self, topology):
        #input and output layers must be the same size
        assert(topology[0] == topology[-1]);
        Net.__init__(topology)

    def train(self, inputVals):
        Net.feedForward(inputVals)
        Net.backProp(inputVals) #this is the only difference, backProp using input values rather than any outputs


if __name__ == '__main__':
    topology = [3, 2, 3]
    encoder = AutoEncoder(topology)
