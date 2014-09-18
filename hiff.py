import numpy as np
class HIFF(object):
	"""
	Hierarchical If and Only If function
	"""
	def __init__(self, NUMGENES=16,K=2,P=4,Rc=2,fixed_mask=False):
		super(HIFF, self).__init__()
		self.NUMGENES=NUMGENES
		self.K = K
		self.P = P
		self.Rc = Rc
		self.NONVALUE = -1
		self.order = [i for i in range(0,NUMGENES)]
		if fixed_mask:
			self.mask = self.get_fixed_mask(NUMGENES)
		else:
			# random mask
			self.mask = np.random.binomial(1,0.5,NUMGENES)

	def H(self,_genes):
		genes = _genes^self.mask
		bonus = 1
		F = 0
		transform = [0]*len(genes)
		last = len(genes)
		for b in range(0,last):
			transform[b]=genes[self.order[b]]
			F+=self.f(transform[b])*bonus
		for level in range(1,self.P+1):
			last=last//self.K
			bonus = bonus * self.Rc
			for b in range(0,last):
				transform[b]=self.t(transform,b*self.K)
				F+=self.f(transform[b])*bonus
		return F

	def t(self,transform,first):
		s = 0
		for i in range(first+1,first+self.K):
			s += (transform[first]==transform[i])
		if (s == (self.K-1)):
			return transform[first]
		return self.NONVALUE

	def f(self,b):
		if b != self.NONVALUE:
			return 1
		return 0

	def get_fixed_mask(self,NUMGENES):
		if NUMGENES == 64:
			return np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1,
					       0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0,
					       1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])

		elif NUMGENES == 128:
			return np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
					       1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
					       0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
					       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
					       0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1,
					       0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
if __name__ == '__main__':
	h = HIFF(NUMGENES=64,K=2,P=6)
	print h