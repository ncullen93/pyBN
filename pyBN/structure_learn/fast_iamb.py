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

import numpy as np
from pyBN.independence.independence_tests import are_independent

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
	pass