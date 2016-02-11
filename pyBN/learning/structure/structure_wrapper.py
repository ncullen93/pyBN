"""
Wrapper for Structure Learning.

NOTE: This is not currently updated.
"""

from pyBN.structure_learn import *


def learn_structure(data, method, target=None, feature_selection=None):
	"""
	Wrapper for Bayesian network structure learning.

	Arguments
	---------
	*data* : a nested numpy array

	*method* : a string
		Which structure learning algorithm to use.

	*feature_selection* : an integer (optional - default "None")
		If *feature_selection* argument is supplied with an integer,
		it will assume that structure learning is
		being used for feature selection only - so the 
		markov blanket of the *feature_selection* passed-in value will
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

	elif method == 'TAN' or method == 'tan':
		bn = TAN(data, target)

	elif method == 'fast_iamb':
		bn = fast_iamb(data, feature_selection) # learn structure

	elif method == 'gs' or method == 'grow_shrink':
		bn = gs(data, feature_selection)

	elif method == 'iamb':
		bn = iamb(data, feature_selection)

	elif method == 'lambda_iamb':
		bn = lambda_iamb(data, feature_selection)

	elif method == 'nb' or method == 'naive_bayes':
		bn = naive_bayes(data)

	elif method == 'pc' or method == 'path_condition':
		bn = pc(data)
	
	else:
		bn = gs(data) #default

	return bn