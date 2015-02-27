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
