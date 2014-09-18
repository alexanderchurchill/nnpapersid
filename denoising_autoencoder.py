import cPickle
import gzip
import time
#import PIL.Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

#from utils import tile_raster_images
from logistic_sgd import load_data
import pdb

class dA(object):
    """
    Denoising Autoencoders
    """ 
    def __init__(self, input=None, n_visible=784, n_hidden=500,
        W=None, hbias=None, vbias=None, numpy_rng=None,theano_rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                      dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.bh = hbias
        self.bv = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.bh, self.bv]


    def get_corrupted_input(self, input, corruption_level):
       """ This function keeps ``1-corruption_level`` entries of the inputs the same
       and zero-out randomly selected subset of size ``coruption_level``
       Note : first argument of theano.rng.binomial is the shape(size) of
              random numbers that it should produce
              second argument is the number of trials
              third argument is the probability of success of any trial

               this will produce an array of 0s and 1s where 1 has a probability of
               1 - ``corruption_level`` and 0 with ``corruption_level``
       """
       mask = self.theano_rng.binomial(size=input.shape, n=1, p=corruption_level,dtype='int8') 
       input_int = theano.tensor.cast(input,'int8')
       #mask = theano.printing.Print('Printing the mask')(mask)
       #idx = self.mask[:,:]==1
       #idx = theano.printing.Print()(idx)
       input_flip = input_int^mask
       #input_flip = theano.printing.Print('Printing flipped input')(input_flip)
       return input_flip

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.bh)

    def get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        return  T.nnet.sigmoid(T.dot(hidden, self.W.T) + self.bv)

    def build_dA(self,corruption_level):
        self.tilde_input = self.get_corrupted_input(self.input, corruption_level)
        self.h = self.get_hidden_values(self.tilde_input)
        self.z = self.get_reconstructed_input(self.h)
        self.L = -T.sum(self.input * T.log(self.z) + (1 - self.input) * T.log(1 - self.z), axis=1)
        self.l1 = abs(self.W).sum()
        self.l2 = abs(self.W**2).sum()
        self.cost = T.mean(self.L) + 0.0001*self.l2
        self.sample = self.theano_rng.binomial(size=self.input.shape,n=1,p=self.z)
    


