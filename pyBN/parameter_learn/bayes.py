"""
*******************
Bayesian Estimation
Parameter Learning
*******************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np


def bayes_estimator(bn, data, sample_size, prior_dict=None):
	"""
	Bayesian Estimation method of parameter learning.
	This method proceeds by either 1) assuming a uniform prior
	over the parameters based on the Dirichlet distribution
	with an equivalent sample size = *sample_size*, or
	2) assuming a prior as specified by the user with the 
	*prior_dict* argument. The prior distribution is then
	updated from observations in the data based on the
	Multinomial distribution - for which the Dirichlet
	is a "conjugate prior."

	Note that the Bayesian and MLE estimators essentially converge
	to the same set of values as the size of the dataset increases.

	Also note that, unlike the structure learning algorithms, the
	parameter learning functions REQUIRE a passed-in BayesNet object
	because there MUST be some pre-determined structure for which
	we can actually learn the parameters. You can't learn parameters
	without structure - so structure must always be there first!

	Arguments
	---------
	*data* : a nested numpy array
		Data from which to learn parameters

	*sample_size* : an integer
		The "equivalent sample size" (see function summary)

	*prior_dict* : a dictionary, where key = random variable
		and for each key the value is another dictionary where
		key = an instantiation for the random variable and the
		value is its FREQUENCY (an integer value, NOT its relative
		proportion/probability).
	
	Returns
	-------
	None

	Effects
	-------
	- modifies/sets bn.data to the learned parameters

	Notes
	-----

	"""
	pass













