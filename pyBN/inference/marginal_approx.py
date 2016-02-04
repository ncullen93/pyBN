"""
******************************
Approximate Marginal Inference
******************************

Perform approx. marginal inference over a BayesNet object,
with or without evidence.

Eventually, there will be a wrapper function "marginal_approx"
for all of the algorithms, and users can choose their method as
an argument to that function.

Approximate Marginal Inference Algorithms
-----------------------------------------
	
	- Forward Sampling
	- Likelihood Weighted Sampling
	- Gibbs (MCMC) Sampling
	- Loopy Belief Propagation

References
----------
[1] Koller, Friedman (2009). "Probabilistic Graphical Models."

"""

__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor 
from pyBN.utils.graph import topsort

import numpy as np


def marginal_fs_a(bn, n=1000):
	"""
	Approximate marginal probabilities from
	forward sampling algorithm on a BayesNet object.

	This algorithm works by
	repeatedly sampling from the BN and taking
	the ratio of observations as their marginal probabilities.

	One sample is done by first sampling from any prior random
	variables, then moving down the network in topological sort
	order - sampling from each successive random variable by
	conditioning on its parents (which have already been sampled
	higher up the network).

	Note that there is no evidence to include in this algorithm - 
	the comparative algorithm which includes evidence is the 
	likelihood weighted algorithm (see "lw_sample" function).

	Arguments
	---------
	*bn* : a BayesNet object

	*n* : an integer
		The number of samples to take

	Returns
	-------
	*sample_dict* : a dictionary, where key = rv, value = another dict
					where key = instance, value = its probability value

	Notes
	-----
	- Evidence is not currently implemented.
	"""

	sample_dict = {}
	for var in bn.nodes():
	    sample_dict[var] = {}
	    for val in bn.values(var):
	        sample_dict[var][val] = 0

	for i in range(n):
	    #if i % (n/float(10)) == 0:
	     #   print 'Sample: ' , i
	    new_sample = {}
	    for rv in bn.nodes():
	        f = Factor(bn,rv)
	        for p in bn.parents(rv):
	            f.reduce_factor(p,new_sample[p])
	        choice_vals = bn.values(rv)
	        choice_probs = f.cpt
	        chosen_val = np.random.choice(choice_vals, p=choice_probs)

	        sample_dict[rv][chosen_val] += 1
	        new_sample[rv] = chosen_val

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] = int(sample_dict[rv][val]) / float(n)
	
	return sample_dict

def marginal_lws_a(bn, evidence={}, target=None, n=1000):
	"""
	Approximate Marginal probabilities from
	likelihood weighted sample algorithm on
	a BayesNet object.

	Arguments
	---------
	*bn* : a BayesNet object

	*n* : an integer
		The number of samples to take

	*evidence* : a dictionary, where
		key = rv, value = instantiation

	Returns
	-------
	*sample_dict* : a dictionary where key = rv
		and value = another dictionary where
		key = rv instantiation and value = marginal
		probability

	Effects
	-------
	None

	Notes
	-----

	"""
	sample_dict = {}
	weight_list = np.ones(n)

	#factor_dict = dict([(var,Factor(bn, var)) for var in bn.V])
	#parent_dict = dict([(var, bn.data[var]['parents']) for var in bn.V])

	for var in bn.nodes():
	    sample_dict[var] = {}
	    for val in bn.values(var):
	        sample_dict[var][val] = 0

	for i in range(n):
	    #if i % (n/float(10)) == 0:
	     #   print 'Sample: ' , i
	    new_sample = {}
	    for rv in bn.nodes():
	        f = Factor(bn,rv)
	        # reduce_factor by parent samples
	        for p in bn.parents(rv):
	            f.reduce_factor(p,new_sample[p])
	        # if rv in evidence, choose that value and weight
	        if rv in evidence:
	            chosen_val = evidence[rv]
	            weight_list[i] *= f.cpt[bn.values(rv).index(evidence[rv])]
	        # if rv not in evidence, sample as usual
	        else:
	            choice_vals = bn.values(rv)
	            choice_probs = f.cpt
	            chosen_val = np.random.choice(choice_vals, p=choice_probs)
	            
	        new_sample[rv] = chosen_val
	    # weight the choice by the evidence likelihood    
	    for rv in new_sample:
	        sample_dict[rv][new_sample[rv]] += 1*weight_list[i]

	weight_sum = sum(weight_list)

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] /= weight_sum
	        sample_dict[rv][val] = round(sample_dict[rv][val],4)
	
	if target is not None:
		return sample_dict[target]
	else:
		return sample_dict


def marginal_gs_a(bn, n=1000, burn=200):
	"""
	Approximate Marginal probabilities from Gibbs Sampling
	over a BayesNet object.

	Arguments
	---------
	*bn* : a BayesNet object

	*n* : an integer
		The number of samples to take

	*burn* : an integer
		The number of beginning samples to
		throw away for the MCMC mixing.

	Returns
	-------
	*sample_dict* : a dictionary where key = rv
		and value = another dictionary where
		key = rv instantiation and value = marginal
		probability

	Notes
	-----

	"""

	sample_dict ={}
	for rv in bn.nodes():
	    sample_dict[rv]={}
	    for val in bn.values(rv):
	        sample_dict[rv][val] = 0

	state = {}
	for rv in bn.nodes():
	    state[rv] = np.random.choice(bn.values(rv)) # uniform sample

	for i in range(n):
	    #if i % (n/float(10)) == 0:
	      #  print 'Sample: ' , i
	    for rv in bn.nodes():
	        # get possible values conditioned on everything else
	        parents = bn.parents(rv)
	        # no parents - prior
	        if len(parents) == 0:
	            choice_vals = bn.values(rv)
	            choice_probs = bn.cpt(rv)
	        # has parent - filter cpt
	        else:
	            f = Factor(bn,rv)
	            for p in parents:
	                f.reduce_factor(p,state[p])
	            choice_vals = bn.values(rv)
	            choice_probs = f.cpt
	        # sample over remaining possibilities
	        chosen_val = np.random.choice(choice_vals,p=choice_probs)
	        state[rv]=chosen_val
	    # update sample_dict dictionary
	    if i > burn:
	        for rv,val in state.items():
	            sample_dict[rv][val] +=1

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] = round(int(sample_dict[rv][val]) / float(n-burn),4)
	
	return sample_dict


def marginal_lbp_a(target=None, evidence=None, max_iter=100):
	"""
	Perform Message Passing (Loopy Belief Propagation) 
	over a cluster graph.

	This is a good candidate for Numba JIT.

	See Koller pg. 397.


	Parameters
	----------


	Returns
	-------


	Notes
	-----
	- Copied from clustergraph class... not tested.
	- Definitely a problem due to normalization (prob vals way too small)
	- Need to check the scope w.r.t. messages.. all clusters should not
	be accumulating rv's in their scope over the course of the algorithm.
	"""
	def collect_beliefs(cgraph):
		cgraph.beliefs = {}
		for cluster in self.V:
			cgraph.V[cluster].collect_beliefs()
			#print 'Belief ' , cluster , ' : \n', self.V[cluster].belief.cpt
			cgraph.beliefs[cluster] = cgraph.V[cluster].belief

	# 1: Moralize the graph
	# 2: Triangluate
	# 3: Build a clique tree using max spanning
	# 4: Propagation of probabilities using message passing

	# creates clique tree and assigns factors, thus satisfying steps 1-3
	cgraph = ClusterGraph(bn)

	edge_visit_dict = dict([(i,0) for i in cgraph.E])

	iteration = 0
	while not cgraph.is_calibrated():
		if iteration == max_iter:
			break
		if iteration % 50 == 0:
			print 'Iteration: ' , iteration
			for cluster in cgraph.V.values():
				cluster.collect_beliefs()
		# select an edge
		e_idx = np.random.randint(0,len(cgraph.E))
		edge_select = cgraph.E[e_idx]
		p_idx = np.random.randint(0,2)
		parent_edge = edge_select[p_idx]
		child_edge = edge_select[np.abs(p_idx-1)]
		print parent_edge , child_edge

		# send a message along that edge
		cgraph.V[parent_edge].send_message(cgraph.V[child_edge])

		iteration += 1
	print 'Now Collecting Beliefs..'
	collect_beliefs(cgraph)
	#bn.ctree = self








