import cPickle
import numpy
import theano
import theano.tensor as T
import os
from theano.tensor.shared_randomstreams import RandomStreams
import pdb

class Deep_dA(object):
    """
    Denoising Autoencoders
    """ 
    def __init__(self, input=None, n_visible=784, n_hidden=[500,500],
        W=None, hbias=None, vbias=None, numpy_rng=None,theano_rng=None,squared_err=False):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_layers = len(n_hidden)
        self.squared_err = squared_err
        assert (self.n_layers > 0)
        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng
        self.input = input
        self.W = []
        self.bh = []
        self.bv = []
        if not input:
            self.input = T.matrix('input')

        for i in xrange(self.n_layers):
          if i == 0:
            input_size = self.n_visible
          else:
            input_size = self.n_hidden[i-1]
          output_size = self.n_hidden[i]
          initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (input_size + output_size)),
                      high=4 * numpy.sqrt(6. / (input_size + output_size)),
                      size=(input_size, output_size)),
                      dtype=theano.config.floatX)
          W = theano.shared(value=initial_W, name='W_%d'%(i), borrow=True)
          self.W.append(W)
          hbias = theano.shared(value=numpy.zeros(output_size,dtype=theano.config.floatX),
                                name='bh_%d'%(i), borrow=True)
          self.bh.append(hbias)
          vbias = theano.shared(value=numpy.zeros(input_size,dtype=theano.config.floatX),
                                name='bv_%d'%(self.n_layers-1-i), borrow=True)
          self.bv.append(vbias)

        self.params = self.W + self.bh + self.bv


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

    def forward_pass(self,inputs):
        h = []
        for i in xrange(self.n_layers):
          if i==0:
            h.append(T.nnet.sigmoid(T.dot(inputs, self.W[i]) + self.bh[i]))
          else:
            h.append(T.nnet.sigmoid(T.dot(h[-1], self.W[i]) + self.bh[i]))
        return h[-1]

    def reconstruction_pass(self,inputs):
        o = []
        for i in xrange(self.n_layers):
          index = len(self.n_hidden)-1-i
          if i==0:
            o.append(T.nnet.sigmoid(T.dot(inputs, self.W[index].T) + self.bv[index]))
          else:
            o.append(T.nnet.sigmoid(T.dot(o[-1], self.W[index].T) + self.bv[index]))
        return o[-1]

    def build_dA(self,corruption_level):
        self.corruption_level = corruption_level
        self.tilde_input = self.get_corrupted_input(self.input, corruption_level)
        self.h = self.forward_pass(self.tilde_input)
        self.z = self.reconstruction_pass(self.h)
        #self.l1 = abs(self.W).sum()
        self.l2 = 0.
        for i in range(self.n_layers):
          self.l2 += abs(self.W[i]**2).sum()
        if not self.squared_err:
          print 'Building binary cross-entropy cost function.'
          self.L = -T.sum(self.input * T.log(self.z) + (1 - self.input) * T.log(1 - self.z), axis=1)
          self.cost = T.mean(self.L) + 0.001*self.l2
          self.sample = self.theano_rng.binomial(size=self.input.shape,n=1,p=self.z)
        else:
          print 'Building squared_err.'
          self.L = T.sum((self.input-self.z)**2,axis=1)
          self.cost = T.mean(self.L) + 0.0000*self.l2 #Should we keep this?
          self.sample = self.z

    def build_sampler(self,k=20):
        samples,updates = theano.scan(lambda v:self.one_step(v),outputs_info=[self.input],
                                      n_steps=k)
        return samples,updates

    def one_step(self,input):
        input_tilde = self.get_corrupted_input(input,self.corruption_level)
        h = self.forward_pass(input_tilde)
        out = self.reconstruction_pass(h)
        sample = self.theano_rng.binomial(size=self.input.shape,n=1,p=out)
        sample = T.cast(sample,dtype=theano.config.floatX)
        return sample


class Deep_dA_cont(Deep_dA):
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
       mask = self.theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level,dtype=theano.config.floatX) 
       input_flip = input*mask
       #input_flip = theano.printing.Print('Printing flipped input')(input_flip)
       return input_flip


if __name__ == '__main__':

    dda = Deep_dA(n_visible=500,n_hidden=[100,])
    dda.build_dA(0.1)
    pdb.set_trace()
    


