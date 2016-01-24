"""
*****************************
Maximum Likelihood Estimation
Parameter Learning
*****************************

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

def mle_estimator(bn, data):
	"""
	Maximum Likelihood Estimation is a frequentist
	method for parameter learning, where there is NO
	prior distribution. Instead, the frequencies/counts
	for each parameter start at 0 and are simply incremented
	as the relevant parent-child values are observed in the
	data. 

	This can be a risky method for small datasets, because if a 
	certain parent-child instantiation is never observed in the
	data, then its probability parameter will be ZERO (even if you
	know it should at least have a very small probability). 

	Note that the Bayesian and MLE estimators essentially converge
	to the same set of values as the size of the dataset increases.

	Also note that, unlike the structure learning algorithms, the
	parameter learning functions REQUIRE a passed-in BayesNet object
	because there MUST be some pre-determined structure for which
	we can actually learn the parameters. You can't learn parameters
	without structure - so structure must always be there first!

	Arguments
	---------
	*bn* : a BayesNet object
		The associated network structure for which
		the parameters will be learned

	*data* : a nested numpy array

	Returns
	-------
	None

	Effects
	-------
	- modifies/sets bn.data to the learned parameters

	Notes
	-----
	data attributes:
		"numoutcomes" : an integer
	    "vals" : a list
	    "parents" : a list or None
	    "children": a list or None
	    "cprob" : a nested python list

	"""
	# map edge_list with strings to edge_idx with integer columns of each vertex
	# these are the parent-child combinations to check in each data observation
	# we could leave these as strings if the data was in a pandas dataframe
	edge_idx = [([bn.V.index(p) for p in bn.data[rv]['parents']],bn.V.index(rv)) for rv in bn.V]
	# set empty conditional probability table for each RV
	for rv in bn.V:
		p_idx = int(np.prod([len(bn.data[p]['vals']) for p in bn.data[rv]['parents']]))
		bn.data[rv]['cprob'] = [[0]*len(bn.data[rv]['vals']) for _ in range(p_idx)]

	# loop through each row of data
	for row in data:
		# loop through each edge in edge_idx
		for parents,rv in edge_idx:
			obs = np.empty((len(parent)+1,))
			# observe the instantiation of parents and rv
			obs[0] = bn.data[rv]['vals'].index(row[rv]) # get rv instance
			idx = 1
			for p in parents:
				obs[idx] = bn.data[p]['vals'].index(row[p]) # get parent instance
				idx+=1
			# cprob is always a 2D array if num parents > 0
			# figure out which idx combination of parents it is
			# need to use cardinalities and strides..
			
			bn.data[rv]['cprob'][obs[0]][parent_idx] += 1

	














