"""
******************
IAMB STRUCTURE
LEARNING ALGORITHM
******************

"""

import numpy as np
from pyBN.utils.independence_tests import are_independent
from pyBN.utils.orient_edges import orient_edges_MB

def iamb(data, alpha=0.05):
	"""
	IAMB Algorithm for learning the structure of a
	Discrete Bayesian Network from data.

	Arguments
	---------
	*data* : a nested numpy array

	*dag* : a boolean
		Whether to return a Directed graph

	*pdag* : a boolean
		Whether to return a Partially Directed Graph.

	*alpha* : a float
		The type II error rate.

	Returns
	-------
	*bn* : a BayesNet object

	Effects
	-------
	None

	Notes
	-----
	"""
	n_rv = data.shape[1]
	Mb = dict([(rv,{}) for rv in range(n_rv)])
	

	if fs is None:
		_T = range(n_rv)
	else:
		assert (not isinstance(fs, list)), 'fs must be only one value'
		_T = [fs]

	# LEARN MARKOV BLANKET
	for T in _T:

		V = set(range(n_rv)) - {T}
		Mb_change=True

		# GROWING PHASE
		while Mb_change:
			Mb_change = False
			# find X_max in V-Mb(T)-{T} that maximizes 
			# mutual information of X,T|Mb(T)
			# i.e. max of mi_test(data[:,(X,T,Mb(T))])
			max_val = -1
			max_x = None
			for X in V - Mb(T) - {T}:
				cols = (X,T)+tuple(Mb(T))
				mi_val = mi_test(data[:,cols],test=False)
				if mi_val > max_val:
					max_val = mi_val
					max_x = X
			# if Xmax is dependent on T given Mb(T)
			cols = (max_x,T) + tuple(Mb(T))
			if are_independent(data[:,cols]):
				Mb(T).add(X)
				Mb_change = True

		# SHRINKING PHASE
		for X in Mb(T):
			# if x is indepdent of t given Mb(T) - {x}
			cols = (X,T) + tuple(Mb(T)-{X})
			if are_independent(data[:,cols],alpha):
				Mb(T).remove(X)

	if fs is None:
		# RESOLVE GRAPH STRUCTURE
		edge_dict = resolve_markov_blanket(Mb, data)

		# ORIENT EDGES
		oriented_edge_dict = orient_edges_MB(edge_dict,Mb,data,alpha)

		# CREATE BAYESNET OBJECT
		value_dict = dict(zip(range(data.shape[1]),
			[list(np.unique(col)) for col in data.T]))
		bn=BayesNet(oriented_edge_dict,value_dict)

		return bn
	else:
		return Mb[_T]
























