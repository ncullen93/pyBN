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

from scipy.stats import chi2_contingency

def pearson_chi2(X, Y, U=None):
	"""
	Perform Pearson's Chi-Squared test for independence.

	Arguments
	---------
	*X* : a numpy array
	*U* : a nested numpy array ?
		The set of random variables we condition upon.. meaning that
		we fix the value of the rv(s) in U and then test for correlation.

	"""
	#correlation = sp.stats.pearsonr(X,Y)
	correlation = chi2_contingency(X,Y)
	return(correlation)
