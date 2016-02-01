"""
Wrapper for Structure Learning
"""

from pyBN.structure_learn import *


def learn_structure(data, method, fs=None):
	"""
	Wrapper for Bayesian network structure learning.

	Arguments
	---------
	*data* : a nested numpy array

	*method* : a string
		Which structure learning algorithm to use.

	*fs* : an integer (optional - default "None")
		If *fs* argument is supplied with an integer,
		it will assume that structure learning is
		being used for feature selection only - so the 
		markov blanket of the *fs* passed-in value will
		be calculated and returned. Note, this is only
		relevant for a few constraint-based structure
		learning algorithms

	Returns
	-------
	*bn* : a structure learned BayesNet object

	"""
	### LEARN STRUCTURE OF BN ###
	if method == 'chow_liu':
		bn = chow_liu(data)

	elif method == 'fast_iamb':
		bn = fast_iamb(data, fs) # learn structure

	elif method == 'gs' or method == 'grow_shrink':
		bn = gs(data, fs)

	elif method == 'iamb':
		bn = iamb(data, fs)

	elif method == 'lambda_iamb':
		bn = lambda_iamb(data, fs)

	elif method == 'nb' or method == 'naive_bayes':
		bn = naive_bayes(data)

	elif method == 'pc' or method == 'path_condition':
		bn = pc(data)
	
	else:
		bn = gs(data) #default

	return bn