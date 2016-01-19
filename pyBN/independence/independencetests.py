"""
******************************
Conditional Independence Tests
******************************

Constraint-based Tests
----------------------
- mutual information (parametric, semiparametric and permutation tests)
- shrinkage-estimator for the mutual information
- Pearson's X^2 (parametric, semiparametric and permutation tests)

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
import pandas as pd
from scipy import stats

def mutual_information(data):
	"""
	We use the maximum likelihood estimators as probabilities. The
	cross entropy (mutual information) is computed, then the 
	chi square test is used, with degrees of freedom equal to 
	(|X|-1)*(|Y|-1)*Pi_z\inZ(|z|)

	Steps:
		- Calculate the marginal/conditional probabilities
		- Compute the Cross Entropy (Mutual Information) value
		- Compute the degrees of freedom
		- Compute the chi square test

	Arguments
	----------
	*data* : a pandas dataframe or numpy array
	*X* : a string or an integer
		X column
	*Y* : a string or an integer
		Y column
	*Z* : a string/integer, or a list/tuple of strings/integers
		Conditioned-upon variables

	Returns
	-------
	*p_val* : a float
		The result of the chi square test

	Effects
	-------
	None

	Notes
	-----
	- Assuming for now that |Z| = 1... generalize later
	"""
	#data = np.array(data) # create data
	#bins = (5,5,5) # create bins
	hist,_ = np.histogramdd(data, bins=(2,2,2)) # frequency counts

	Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z

	Px = np.sum(Pxyz, axis = (1,2),keepdims=True) # P(X)
	Py = np.sum(Pxyz, axis = (0,2),keepdims=True) # P(Y)
	Pz = np.sum(Pxyz, axis = (0,1),keepdims=True) # P(Z)

	Pxy = np.sum(Pxyz, axis = 2,keepdims=True) # P(X,Y)
	Pxz = np.sum(Pxyz, axis = 1,keepdims=True) # P(X,Z)
	Pyz = np.sum(Pxyz, axis = 0,keepdims=True) # P(Y,Z)
	

	Pxy_z = Pxyz / Pz # P(X,Y | Z) = P(X,Y,Z) / P(Z)
	Px_z = Pxz / Pz # P(X | Z) = P(X,Z) / P(Z)	
	Py_z = Pyz / Pz # P(Y | Z) = P(Y,Z) / P(Z)

	MI = np.sum(Pxyz * np.log(Pxy_z / Px_z*Py_z))

	return(MI)

def condind(data):
	"""
	Test null hypothesis that P(X,Y,Z) = P(Z)P(X|Z)P(Y|Z)
	versus empirically observed P(X,Y,Z) in the data

	Steps
		- Calculate P(XYZ) empirically and expected
		- Compute ddof
		- Perfom one-way chisquare to test
	"""
	hist,_ = np.histogramdd(data,bins=(2,2,2))

	# calculate
	Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z

	Px = np.sum(Pxyz, axis = (1,2),keepdims=True) # P(X)
	Py = np.sum(Pxyz, axis = (0,2),keepdims=True) # P(Y)
	Pz = np.sum(Pxyz, axis = (0,1),keepdims=True) # P(Z)

	Pxy = np.sum(Pxyz, axis = 2) # P(X,Y)
	Pxz = np.sum(Pxyz, axis = 1) # P(X,Z)
	Pyz = np.sum(Pxyz, axis = 0) # P(Y,Z)
	

	Pxy_z = Pxyz / Pz # P(X,Y | Z) = P(X,Y,Z) / P(Z)
	Px_z = Pxz / Pz # P(X | Z) = P(X,Z) / P(Z)	
	Py_z = Pyz / Pz # P(Y | Z) = P(Y,Z) / P(Z)

	observed_dist = Pxyz # Empirical distribution
	expected_dist = Px*(Py_z*Px_z) #  Conditionally independent null distribution
	expected_dist /= expected_dist.sum()

	observed = observed_dist.flatten() * len(data)
	expected = expected_dist.flatten() * len(data)

	chi2, p_val = stats.chisquare(observed,expected, ddof=data.shape[1]-1)

	return chi2, p_val





