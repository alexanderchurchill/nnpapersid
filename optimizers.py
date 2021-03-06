# Author: Nicolas Boulanger-Lewandowski
# University of Montreal, 2012


import numpy, sys
import theano
import theano.tensor as T
import cPickle
import os
import pdb
import copy


def sgd_optimizer(p,inputs,costs,train_set,updates_old=None,monitor=None,consider_constant=[],lr=0.001,
                  num_epochs=300,save=False,output_folder=None,iteration=0):
  '''SGD optimizer with a similar interface to hf_optimizer.
  p: list of params wrt optimization is performed
  cost: theano scalar defining objective function
  inputs: [list] of symbolic inputs to graph. Must include targets if supervised
  updates_old: The updates dictionary for the sharedvariables. Check scan documentation for details.
  monitor: Monitoring cost. If empty, optimization cost is printed.
  consider_constant: Input to T.grad. Check RBM code for example.
  train_set: Dataset in the form of SequenceDataset
  lr: Learning rate for SGD
  '''
  best_cost = numpy.inf
  g = T.grad(costs,p,consider_constant=consider_constant)
  updates = dict((i, i - lr*j) for i, j in zip(p, g))
  if updates_old:
    updates_new = copy.copy(updates_old)
    updates_new.update(updates) 
  else:
    updates_new = {}
    updates_new.update(updates)
  if monitor:
    f = theano.function(inputs, monitor, updates=updates_new)
  else:
    f = theano.function(inputs, costs, updates=updates_new)
  
  try:
    for u in xrange(num_epochs):
      cost = []
      for i in train_set.iterate(True): 
        cost.append(f(*i))
      print 'update %i, cost=' %(u+1), numpy.mean(cost, axis=0)
      this_cost = numpy.absolute(numpy.mean(cost, axis=0))
      if this_cost < best_cost:
        best_cost = this_cost
        print 'Best Params!'
        if save:
          best_params = [i.get_value().copy() for i in p]
          if not output_folder:
            cPickle.dump(best_params,open('best_params_{0}.pickle'.format(iteration),'w'))
          else:
            if not os.path.exists(output_folder):
              os.makedirs(output_folder)
            save_path = os.path.join(output_folder,'best_params_{0}.pickle'.format(iteration))
            cPickle.dump(best_params,open(save_path,'w'))
      sys.stdout.flush()

  except KeyboardInterrupt: 
    print 'Training interrupted.'

class hf_optimizer:
  '''Black-box Theano-based Hessian-free optimizer.
See (Martens, ICML 2010) and (Martens & Sutskever, ICML 2011) for details.

Useful functions:
__init__ :
    Compiles necessary Theano functions from symbolic expressions.
train :
    Performs HF optimization following the above references.'''

  def __init__(self, p, inputs, s, costs, h=None):
    '''Constructs and compiles the necessary Theano functions.

  p : list of Theano shared variables
      Parameters of the model to be optimized.
  inputs : list of Theano variables
      Symbolic variables that are inputs to your graph (they should also
      include your model 'output'). Your training examples must fit these.
  s : Theano variable
    Symbolic variable with respect to which the Hessian of the objective is
    positive-definite, implicitly defining the Gauss-Newton matrix. Typically,
    it is the activation of the output layer.
  costs : list of Theano variables
      Monitoring costs, the first of which will be the optimized objective.
  h: Theano variable or None
      Structural damping is applied to this variable (typically the hidden units
      of an RNN).'''

    self.p = p
    self.shapes = [i.get_value().shape for i in p]
    self.sizes = map(numpy.prod, self.shapes)
    self.positions = numpy.cumsum([0] + self.sizes)[:-1]

    g = T.grad(costs[0], p)
    g = map(T.as_tensor_variable, g)  # for CudaNdarray
    self.f_gc = theano.function(inputs, g + costs)  # during gradient computation
    self.f_cost = theano.function(inputs, costs)  # for quick cost evaluation

    symbolic_types = T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4

    coefficient = T.scalar()  # this is lambda*mu    
    if h is not None:  # structural damping with cross-entropy
      h_constant = symbolic_types[h.ndim]()  # T.Rop does not support `consider_constant` yet, so use `givens`
      structural_damping = coefficient * (-h_constant*T.log(h) - (1-h_constant)*T.log(1-h)).sum() / h.shape[0]
      costs[0] += structural_damping
      givens = {h_constant: h}
    else:
      givens = {}

    # this computes the product Gv = J'HJv (G is the Gauss-Newton matrix)
    v = [symbolic_types[len(i)]() for i in self.shapes]
    Jv = T.Rop(s, p, v)
    HJv = T.grad(T.sum(T.grad(costs[0], s,disconnected_inputs='ignore' )*Jv), s, consider_constant=[Jv],disconnected_inputs='ignore' )
    Gv = T.grad(T.sum(HJv*s), p, consider_constant=[HJv, Jv])
    Gv = map(T.as_tensor_variable, Gv)  # for CudaNdarray
    self.function_Gv = theano.function(inputs + v + [coefficient], Gv, givens=givens,
                                       on_unused_input='ignore')

  def quick_cost(self, delta=0):
    # quickly evaluate objective (costs[0]) over the CG batch
    # for `current params` + delta
    # delta can be a flat vector or a list (else it is not used)
    if isinstance(delta, numpy.ndarray):
      delta = self.flat_to_list(delta)

    if type(delta) in (list, tuple):
      for i, d in zip(self.p, delta):
        i.set_value(i.get_value() + d)

    cost = numpy.mean([self.f_cost(*i)[0] for i in self.cg_dataset.iterate(update=False)])

    if type(delta) in (list, tuple):
      for i, d in zip(self.p, delta):
        i.set_value(i.get_value() - d)

    return cost


  def cg(self, b):
    if self.preconditioner:
      M = self.lambda_ * numpy.ones_like(b)
      for inputs in self.cg_dataset.iterate(update=False):
        M += self.list_to_flat(self.f_gc(*inputs)[:len(self.p)])**2  #/ self.cg_dataset.number_batches**2
      #print 'precond~%.3f,' % (M - self.lambda_).mean(),
      M **= -0.75  # actually 1/M
      sys.stdout.flush()
    else:
      M = 1.0

    x = self.cg_last_x if hasattr(self, 'cg_last_x') else numpy.zeros_like(b)  # sharing information between CG runs
    r = b - self.batch_Gv(x)
    d = M*r
    delta_new = numpy.dot(r, d)
    phi = []
    backtracking = []
    backspaces = 0

    for i in xrange(1, 1 + self.max_cg_iterations):
      # adapted from http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf (p.51)
      q = self.batch_Gv(d)
      dq = numpy.dot(d, q)
      #assert dq > 0, 'negative curvature'
      alpha = delta_new / dq
      x = x + alpha*d
      r = r - alpha*q
      s = M*r
      delta_old = delta_new
      delta_new = numpy.dot(r, s)
      d = s + (delta_new / delta_old) * d

      if i >= int(numpy.ceil(1.3**len(backtracking))):
        backtracking.append((self.quick_cost(x), x.copy(), i))

      phi_i = -0.5 * numpy.dot(x, r + b)
      phi.append(phi_i)

      progress = ' [CG iter %i, phi=%+.5f, cost=%.5f]' % (i, phi_i, backtracking[-1][0])
      sys.stdout.write('\b'*backspaces + progress)
      sys.stdout.flush()
      backspaces = len(progress)

      k = max(10, i/10)
      if i > k and phi_i < 0 and (phi_i - phi[-k-1]) / phi_i < k*0.0005:
        break

    self.cg_last_x = x.copy()

    if self.global_backtracking:
      j = numpy.argmin([b[0] for b in backtracking])
    else:
      j = len(backtracking) - 1
      while j > 0 and backtracking[j-1][0] < backtracking[j][0]:
        j -= 1
    print ' backtracked %i/%i' % (backtracking[j][2], i),
    sys.stdout.flush()

    return backtracking[j] + (i,)

  def flat_to_list(self, vector):
    return [vector[position:position + size].reshape(shape) for shape, size, position in zip(self.shapes, self.sizes, self.positions)]

  def list_to_flat(self, l):
    return numpy.concatenate([i.flatten() for i in l])

  def batch_Gv(self, vector, lambda_=None):
    v = self.flat_to_list(vector)
    if lambda_ is None: lambda_ = self.lambda_
    result = lambda_*vector  # Tikhonov damping
    for inputs in self.cg_dataset.iterate(False):
      result += self.list_to_flat(self.function_Gv(*(inputs + v + [lambda_*self.mu]))) / self.cg_dataset.number_batches
    return result

  def train(self, gradient_dataset, cg_dataset, initial_lambda=0.1, mu=0.03, global_backtracking=False, preconditioner=False, max_cg_iterations=250, num_updates=100, validation=None, validation_frequency=1, patience=numpy.inf, save_progress=None):
    '''Performs HF training.

  gradient_dataset : SequenceDataset-like object
      Defines batches used to compute the gradient.
      The `iterate(update=True)` method should yield shuffled training examples
      (tuples of variables matching your graph inputs).
      The same examples MUST be returned between multiple calls to iterator(),
      unless update is True, in which case the next batch should be different.
  cg_dataset : SequenceDataset-like object
      Defines batches used to compute CG iterations.
  initial_lambda : float
      Initial value of the Tikhonov damping coefficient.
  mu : float
      Coefficient for structural damping.
  global_backtracking : Boolean
      If True, backtracks as much as necessary to find the global minimum among
      all CG iterates. Else, Martens' heuristic is used.
  preconditioner : Boolean
      Whether to use Martens' preconditioner.
  max_cg_iterations : int
      CG stops after this many iterations regardless of the stopping criterion.
  num_updates : int
      Training stops after this many parameter updates regardless of `patience`.
  validation: SequenceDataset object, (lambda : tuple) callback, or None
      If a SequenceDataset object is provided, the training monitoring costs
      will be evaluated on that validation dataset.
      If a callback is provided, it should return a list of validation costs
      for monitoring, the first of which is also used for early stopping.
      If None, no early stopping nor validation monitoring is performed.
  validation_frequency: int
      Validation is performed every `validation_frequency` updates.
  patience: int
      Training stops after `patience` updates without improvement in validation
      cost.
  save_progress: string or None
      A checkpoint is automatically saved at this location after each update.
      Call the `train` function again with the same parameters to resume
      training.'''

    self.lambda_ = initial_lambda
    self.mu = mu
    self.global_backtracking = global_backtracking
    self.cg_dataset = cg_dataset
    self.preconditioner = preconditioner
    self.max_cg_iterations = max_cg_iterations
    best = [0, numpy.inf, None]  # iteration, cost, params
    first_iteration = 1

    if isinstance(save_progress, str) and os.path.isfile(save_progress):
      save = cPickle.load(file(save_progress))
      self.cg_last_x, best, self.lambda_, first_iteration, init_p = save
      first_iteration += 1
      for i, j in zip(self.p, init_p): i.set_value(j)
      print '* recovered saved model'
    
    try:
      for u in xrange(first_iteration, 1 + num_updates):
        print 'update %i/%i,' % (u, num_updates),
        sys.stdout.flush()

        gradient = numpy.zeros(sum(self.sizes), dtype=theano.config.floatX)
        costs = []
        for inputs in gradient_dataset.iterate(update=True):
          result = self.f_gc(*inputs)
          gradient += self.list_to_flat(result[:len(self.p)]) / gradient_dataset.number_batches
          costs.append(result[len(self.p):])

        print 'cost=', numpy.mean(costs, axis=0),
        print 'lambda=%.5f,' % self.lambda_,
        sys.stdout.flush()

        after_cost, flat_delta, backtracking, num_cg_iterations = self.cg(-gradient)
        delta_cost = numpy.dot(flat_delta, gradient + 0.5*self.batch_Gv(flat_delta, lambda_=0))  # disable damping
        before_cost = self.quick_cost()
        for i, delta in zip(self.p, self.flat_to_list(flat_delta)):
          i.set_value(i.get_value() + delta)
        cg_dataset.update()

        rho = (after_cost - before_cost) / delta_cost  # Levenberg-Marquardt
        #print 'rho=%f' %rho,
        if rho < 0.25:
          self.lambda_ *= 1.5
        elif rho > 0.75:
          self.lambda_ /= 1.5
        
        if validation is not None and u % validation_frequency == 0:
          if validation.__class__.__name__ == 'SequenceDataset':
            costs = numpy.mean([self.f_cost(*i) for i in validation.iterate()], axis=0)
          elif callable(validation):
            costs = validation()
          print 'validation=', costs,
          if costs[0] < best[1]:
            best = u, costs[0], [i.get_value().copy() for i in self.p]
            print '*NEW BEST',

        if isinstance(save_progress, str):
          # do not save dataset states
          save = self.cg_last_x, best, self.lambda_, u, [i.get_value().copy() for i in self.p]
          cPickle.dump(save, file(save_progress, 'wb'), cPickle.HIGHEST_PROTOCOL)
        
        if u - best[0] > patience:
          print 'PATIENCE ELAPSED, BAILING OUT'
          break
        
        print
        sys.stdout.flush()
    except KeyboardInterrupt:
      print 'Interrupted by user.'
    
    if best[2] is None:
      best[2] = [i.get_value().copy() for i in self.p]
    return best[2]






