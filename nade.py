'''
Class that builds the graph for the Neural Autoregressive Distribution Estimator (NADE)
'''


import numpy
import theano
import theano.tensor as T
import pdb

def shared_normal(num_rows, num_cols, scale=1,name=None):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(name=name,value=numpy.random.normal(
    scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))

def shared_zeros(shape,name=None):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(name=name,value=numpy.zeros(shape, dtype=theano.config.floatX))

def sigmoid(x):
  return 0.5*numpy.tanh(0.5*x) + 0.5

class NADE:
  def __init__(self,n_visible,n_hidden):
    self.n_visible = n_visible
    self.n_hidden = n_hidden
    self.W = shared_normal(n_visible, n_hidden, 0.01,'W')
    self.V = shared_normal(n_hidden, n_visible,0.01,'V')
    self.b = shared_zeros(n_visible,)
    self.c = shared_zeros(n_hidden,)
    self.params = self.W,self.V,self.b,self.c
    self.v = T.matrix('v')
    self.s,self.y,self.cost = self.build_NADE(self.v,self.W,self.V,self.b,self.c)
    
  def build_NADE(self,v, W, V, b, c):
    a = T.shape_padright(v) * T.shape_padleft(W)
    a = a.dimshuffle(1, 0, 2)

    c_init = c
    if c.ndim == 1:
        c_init = T.dot(T.ones((v.shape[0], 1)), T.shape_padleft(c))

    (activations, s), updates = theano.scan(lambda V_i, a_i, partial_im1: (a_i + partial_im1, T.dot(V_i, T.nnet.sigmoid(partial_im1.T))), 
                                                   sequences=[V.T, a], outputs_info=[c_init, None])
    s = s.T + b
    y = T.nnet.sigmoid(s)

    cost = -v*T.log(y) - (1-v)*T.log(1-y)
    cost = cost.sum() / v.shape[0]
    return s, y, cost

  def sample(self,):
    result = numpy.empty_like(self.b.get_value())
    a = self.c.get_value()
    for i in xrange(len(self.b.get_value())):
      probability_i = sigmoid(numpy.dot(self.V.get_value()[:, i], sigmoid(a)) + self.b.get_value()[i])
      result[i] = numpy.random.binomial(n=1, p=probability_i)
      a += self.W.get_value()[i, :] * result[i]
    return result

  def sample_clamped(self,sample_dict):
    result = numpy.empty_like(self.b.get_value())
    a = self.c.get_value()
    for i in xrange(len(self.b.get_value())):
      if i in sample_dict:
        result[i] = sample_dict[i]
      else:
        probability_i = sigmoid(numpy.dot(self.V.get_value()[:, i], sigmoid(a)) + self.b.get_value()[i])
      result[i] = numpy.random.binomial(n=1, p=probability_i)
      a += self.W.get_value()[i, :] * result[i]
    return result

  def nade_sample_multi(self, W, V, b, c, n=1):
    result = numpy.empty((n, len(b)))
    probabilities = numpy.empty_like(result)
    a = numpy.resize(c, (n, len(c))).T
    for i in xrange(len(b)):
      probabilities[:, i] = sigmoid(numpy.dot(V[:, i], sigmoid(a)) + b[i])
      result[:, i] = numpy.random.random(size=n) < probabilities[:, i]
      a += numpy.outer(W[i, :], result[:, i])
    return result

  def sample_multiple(self,n=50):
    print "n:",n
    samples = self.nade_sample_multi(self.W.get_value(),self.V.get_value(),self.b.get_value(),self.c.get_value(),n=n)
    return samples

  def load(self,filename):
    params = cPicke.load(filename)
    for i,j in zip(self.params,params):
        i.set_value(j)



if __name__ == '__main__':
  n_visible = 10
  n_hidden = 20
  test = NADE(n_visible,n_hidden)
  gs = T.grad(test.cost,test.params)
  p = test.sample()
  pdb.set_trace()
  print p