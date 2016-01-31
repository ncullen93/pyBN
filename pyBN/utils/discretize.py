"""
**************************
Discretize Continuous Data
**************************

Since pyBN only handles Discrete Bayesian Networks,
and therefore only handles discrete data, it is
important to have effective functions for 
discretizing continuous data. This code aims to
meet that goal.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

def discretize(data, bins=5):
	"""
	Discretize the passed-in dataset. These
	functions will rely on numpy and scipy
	for speed and accuracy... no need to
	reinvent the wheel here. Therefore, pyBN's
	discretization methods are basically just
	wrappers for existing methods.

	Numpy Functions
	---------------
	- np.digitize
	- np.histogram(dd)

	Arguments
	---------
	*data* : a nested numpy array

	Example
	-------
	(From numpy docs):
		>>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
		>>> bins = np.array([0, 5, 10, 15, 20])
		>>> np.digitize(x,bins,right=True)
		array([1, 2, 3, 4, 4])
	"""
	pass