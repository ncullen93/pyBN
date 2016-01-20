"""
******************************
Conditional Independence Tests 
for Constraint-based Learning
******************************

Implemented Constraint-based Tests
----------------------------------
- mutual information
- Pearson's X^2

I may consider putting this code into its own class structure. The
main benefit I could see from doing this would be the ability to 
cache joint/marginal/conditional probabilities for expedited tests.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
import pandas as pd
from scipy import stats

def mi_test(data):
	"""
	We use the maximum likelihood estimators as probabilities. The
	mutual information value is computed, then the 
	chi square test is used, with degrees of freedom equal to 
	(|X|-1)*(|Y|-1)*Pi_z\inZ(|z|).

	This function works on datasets that contain more than three
	columns by concatenating the extra columns into one. For that
	reason, it is a little slower in that case.

	Steps:
		- Calculate the marginal/conditional probabilities
		- Compute the Mutual Information value
		- Calculate chi2 statistic = 2*N*MI
		- Compute the degrees of freedom
		- Compute the chi square p-value

	Speed Comparison (mean - 1000 Loops)
	-----------------------------
	*bnlearn* -> 657 microseconds
		ci.test(lizards, test="mi")
	*pybn* -> 448 microseconds
		mutual_information(lizards)

	Arguments
	----------
	*data* : a nested numpy array
		The data from which to learn - must have at least three
		variables. All conditioned variables (i.e. Z) are compressed
		into one variable.

	Returns
	-------
	*p_val* : a float
		The pvalue from the chi2 and ddof

	Effects
	-------
	None

	Notes
	-----
	- Doesn't currently work with strings... 
	- Should generalize to let data be a Pandas DataFrame --> would
	encourage external use.

	"""
	
	bins = np.amax(data, axis=0) # read levels for each variable
	# CHECK FOR > 3 COLUMNS -> concatenate Z into one column
	if len(bins) > 3:
		data = data.astype('str')
		ncols = len(bins)
		for i in xrange(len(data)):
			data[i,2] = ''.join(data[i,2:ncols])
		data = data.astype('int')[:,0:3]

	hist,_ = np.histogramdd(data, bins=bins[0:3]) # frequency counts

	Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z
	Pz = np.sum(Pxyz, axis = (0,1)) # P(Z)
	Pxz = np.sum(Pxyz, axis = 1) # P(X,Z)
	Pyz = np.sum(Pxyz, axis = 0) # P(Y,Z)	

	Pxy_z = Pxyz / Pz # P(X,Y | Z) = P(X,Y,Z) / P(Z)
	Px_z = Pxz / Pz # P(X | Z) = P(X,Z) / P(Z)	
	Py_z = Pyz / Pz # P(Y | Z) = P(Y,Z) / P(Z)

	Px_y_z = np.empty((Pxy_z.shape)) # P(X|Z)P(Y|Z)
	for i in xrange(bins[0]):
		for j in xrange(bins[1]):
			for k in xrange(bins[2]):
				Px_y_z[i][j][k] = Px_z[i][k]*Py_z[j][k]
	

	MI = np.sum(Pxyz * np.log(Pxy_z / (Px_y_z)))
	chi2_statistic = 2*len(data)*MI
	ddof = (bins[0] - 1) * (bins[1] - 1) * bins[2]
	p_val = 2*stats.chi2.pdf(chi2_statistic, ddof) # 2* for one tail
	return p_val

def chi2_test(data):
	"""
	Test null hypothesis that P(X,Y,Z) = P(Z)P(X|Z)P(Y|Z)
	versus empirically observed P(X,Y,Z) in the data using
	the traditional chisquare test based on observed versus
	expected frequency bins.

	Steps
		- Calculate P(XYZ) empirically and expected
		- Compute ddof
		- Perfom one-way chisquare

	Arguments
	---------
	*data* : a nested numpy array
		The data from which to learn - must have at least three
		variables. All conditioned variables (i.e. Z) are compressed
		into one variable.

	Returns
	-------
	*chi2_statistic* : a float
		Chisquare statistic
	*p_val* : a float
		The pvalue from the chi2 and ddof

	Effects
	-------
	None

	Notes
	-----
	- Assuming for now that |Z| = 1... generalize later
	- Should generalize to let data be a Pandas DataFrame --> would
	encourage external use.

	"""
	# compress extra Z variables at the start.. not implemented yet
	bins = tuple(np.amax(data, axis=0))
	hist,_ = np.histogramdd(data,bins=bins)

	Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z

	Pz = np.sum(Pxyz, axis = (0,1)) # P(Z)
	Pxz = np.sum(Pxyz, axis = 1) # P(X,Z)
	Pyz = np.sum(Pxyz, axis = 0) # P(Y,Z)

	Px_z = Pxz / Pz # P(X | Z) = P(X,Z) / P(Z)	
	Py_z = Pyz / Pz # P(Y | Z) = P(Y,Z) / P(Z)

	observed_dist = Pxyz # Empirical distribution
	# This isn't tested
	expected_dist = Pz # P(Z)P(X|Z)P(Y|Z)
	for i in xrange(bins[0]):
		for j in xrange(bins[1]):
			for k in xrange(bins[2]):
				expected_dist[i][j][k] *= (Px_z[i][k]*Py_z[j][k])

	observed = observed_dist.flatten() * len(data)
	expected = expected_dist.flatten() * len(data)

	ddof = (bins[0] - 1) * (bins[1]- 1) * bins[2]
	chi2_statistic, p_val = stats.chisquare(observed,expected, ddof=ddof)

	return chi2_statistic, p_val





