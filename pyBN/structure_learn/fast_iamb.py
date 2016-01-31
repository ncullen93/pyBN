"""
*********
Fast-IAMB
Algorithm
*********


For Feature Selection (from [1]):
	"A principled solution to the feature selection problem is
	to determine a subset of attributes that can “shield” (render
	independent) the attribute of interest from the effect of
	the remaining attributes in the domain. Koller and Sahami
	[4] first showed that the Markov blanket of a given target attribute
	is the theoretically optimal set of attributes to predict
	its value...
	
	Because the Markov blanket of a target attribute T renders
	it statistically independent from all the remaining attributes
	(see the Markov blanket definition below), all information
	that may influence its value is stored in the values
	of the attributes of its Markov blanket. Any attribute
	from the feature set outside its Markov blanket can be effectively
	ignored from the feature set without adversely affecting
	the performance of any classifier that predicts the
	value of T"

References
----------
[1] Yaramakala and Maragritis, "Speculative Markov Blanket 
Discovery for Optimal Feature Selection"

[2] Tsarmardinos, et al. "Algorithms for Large Scale 
Markov Blanket Discovery" 

"""
from __future__ import division

import numpy as np
from pyBN.independence.independence_tests import are_independent, mi_test

def fast_iamb(data, k=5, alpha=0.05):
	"""
	From [1]:
		"A novel algorithm for the induction of
		Markov blankets from data, called Fast-IAMB, that employs
		a heuristic to quickly recover the Markov blanket. Empirical
		results show that Fast-IAMB performs in many cases
		faster and more reliably than existing algorithms without
		adversely affecting the accuracy of the recovered Markov
		blankets."

	Arguments
	---------
	*data* : a nested numpy array

	*alpha* : a float
		Probability of Type II error

	Returns
	-------
	*bn* : a BayesNet object

	Effects
	-------
	None

	Notes
	-----
	"""
	Mb = dict([(rv,{}) for rv in range(n_rv)])
	n_rv = data.shape[1]
	N = data.shape[0]
	card = dict(zip(range(data.shape[1]),np.amax(data,axis=0)))
	# LEARN MARKOV BLANKET
	for T in xrange(n_rv):
		S = set(range(n_rv)) - {T}
		for A in S:
			if not are_independent(data[:,(A,T)]):
				S.remove(A)
		s_h_dict = dict([(s,0) for s in S])
		while S:
			insufficient_data = False
			break_grow_phase = False
			
			#### GROW PHASE ####
			# Calculate mutual information for all variables
			mi_dict = dict([(s,mi_test(data[:,(s,T)+tuple(Mb[T])])) for s in S])
			for x_i in sorted(mi_dict, key=mi_dict.get,reverse=True):
				# Add top MI-score variables until there isn't enough data for bins
				if (N / card[x_i]*card[T]*np.prod([card[b] for b in Mb(T)])) >= k:
					Mb[T].append(x_i)
				else:
					insufficient_data = True
					break

			#### SHRINK PHASE ####
			removed_vars = False
			for A in Mb[T]:
				cols = (A,T) + tuple(set(Mb[T]) - {A})
				# if A is independent of T given Mb[T], remove A
				if are_independent(data[:,cols]):
					Mb[T].remove(A)
					removed_vars=True

			#### FINALIZE BLANKET FOR "T" OR MAKE ANOTHER PASS ####
			if insufficient_data and not removed_vars:
				break
			else:
				A = set(nodes) - {T} - set(Mb[T])
				S = {}
				for a in A:
					cols = (A,T) + tuple(Mb[T])
					if are_independent(data[:,cols]):
						S.add(a)
	return Mb



























