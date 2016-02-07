"""
*****************************
Maximum Likelihood Estimation
Parameter Learning
*****************************

"""
from __future__ import division

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

def mle_estimator(bn, data, nodes=None):
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

	Finally, note that this function can be used to calculate only
	ONE conditional probability table in a BayesNet object by
	passing in a subset of random variables with the "nodes"
	argument - this is mostly used for score-based structure learning,
	where a single cpt needs to be quickly recalculate after the
	addition/deletion/reversal of an arc.

	Arguments
	---------
	*bn* : a BayesNet object
		The associated network structure for which
		the parameters will be learned

	*data* : a nested numpy array

	*nodes* : a list of strings
		Which nodes to learn the parameters for - if None,
		all nodes will be used as expected.

	Returns
	-------
	None

	Effects
	-------
	- modifies/sets bn.data to the learned parameters

	Notes
	-----
	- Currently doesn't return correct solution

	data attributes:
		"numoutcomes" : an integer
	    "vals" : a list
	    "parents" : a list or None
	    "children": a list or None
	    "cprob" : a nested python list

	- Do not want to alter bn.data directly!

	"""
	if nodes is None:
		nodes = list(bn.nodes())
	else:
		if not isinstance(nodes, list):
			nodes = list(nodes)

	for i, n in enumerate(nodes):
		bn.F[n]['values'] = list(np.unique(data[:,i]))

	obs_dict = dict([(rv,[]) for rv in nodes])
	# set empty conditional probability table for each RV
	for rv in nodes:
		# get number of values in the CPT = product of scope vars' cardinalities
		p_idx = int(np.prod([bn.card(p) for p in bn.parents(rv)])*bn.card(rv))
		bn.F[rv]['cpt'] = [0]*p_idx
	
	# loop through each row of data
	for row in data:
		# store the observation of each variable in the row
		for rv in nodes:
			obs_dict[rv] = row[rv]
		
		#obs_dict = dict([(rv,row[rv]) for rv in nodes])
		# loop through each RV and increment its observed parent-self value
		for rv in nodes:
			rv_dict= { n: obs_dict[n] for n in obs_dict if n in bn.scope(rv) }
			offset = bn.cpt_indices(target=rv,val_dict=rv_dict)[0]
			bn.F[rv]['cpt'][offset]+=1

	
	for rv in nodes:
		cpt = bn.cpt(rv)
		for i in range(0,len(bn.cpt(rv)),bn.card(rv)):
			temp_sum = float(np.sum(cpt[i:(i+bn.card(rv))]))
			for j in range(bn.card(rv)):
				cpt[i+j] /= (temp_sum+1e-7)
				cpt[i+j] = round(cpt[i+j],5)





