"""
*****************************
Maximum Likelihood Estimation
Parameter Learning
*****************************

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""


def mle_estimator(data):
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

	Arguments
	---------
	*data* : a nested numpy array

	Returns
	-------
	Not sure yet

	Effects
	-------
	Not sure yet

	Notes
	-----

	"""
	pass













