import cPickle
import numpy
import theano
import theano.tensor as T
import os
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet.conv import conv2d

import pdb

class conv_AE:
	def __init__(self,input_shape,nb_stack=1,filter_shape=(4,4),n_filters=10,batch_size=20):
		self.input = T.tensor4('input')
		self.n_filters = n_filters
		self._input_shape = input_shape
		self.nb_stack = nb_stack
		self._filter_shape = filter_shape
		self.batch_size = 20
		self.numpy_rng = numpy.random.RandomState(123)
		self.theano_rng = RandomStreams(self.numpy_rng.randint(1234))

		#Inferring all shapes
		self.infer_shapes(self.batch_size)
		#pdb.set_trace()
		initial_W = numpy.asarray(
                    self.numpy_rng.normal(loc=0., scale=0.01,
                   	size = (numpy.prod(self._filter_shape), self.nb_stack, self.n_filters)),
                    dtype=theano.config.floatX).T.reshape(self.filter_shape)

		self.W = theano.shared(value=initial_W, name='W')
		self.h_bias = theano.shared(value=numpy.zeros(n_filters,dtype=theano.config.floatX),name='hbias')
		self.v_bias = theano.shared(value=numpy.zeros((1,nb_stack,1,1),dtype=theano.config.floatX),name='vbias',broadcastable=(True,False,True,True))
		self.params = self.W,self.h_bias,self.v_bias


	
	def infer_shapes(self,batch_size):
		bs = batch_size
		im_sh = self._input_shape
		fil_sh = self._filter_shape
		nb_fil = self.n_filters
		nb_im = self.nb_stack
		self.input_shape = (bs,nb_im) + im_sh
		self.hidden_shape = (bs,nb_fil) + tuple([x+y-1 for x,y in zip(im_sh,fil_sh)])
		self.filter_shape = (nb_fil,nb_im) + fil_sh
		self.filter_shape_rev = (nb_im,nb_fil) + fil_sh

	def fprop(self,v):
		self.hidden_activations = conv2d(v,self.W[:,:,::-1,::-1],image_shape=self.input_shape,
									filter_shape=self.filter_shape,border_model='valid') + self.hbias.dimshuffle('x',0,'x','x')
		self.hidden = T.nnet.sigmoid(self.hidden_activations)

		flip_W = self.W.dimshuffle(1,0,2,3)

		self.output_activations = conv2d(self.hidden,flip_W,image_shape=self.input_shape,
										filter_shape=self.filter_shape_rev,border_model='full') + self.vbias
		self.output = T.nnet.sigmoid(self.output_activations)
		self.sample = self.theano_rng.binomial(size=self.output.shape,n=1,p=self.output,dtype=theano.config.floatX)
		cost = -v*T.log(self.output) - (1-v)*T.log(1-self.output)
    	self.cost = cost.sum() / v.shape[0]
if __name__ == '__main__':
	
	test = conv_AE((1,50),nb_stack=1,filter_shape=(1,10),n_filters=10,batch_size=20)