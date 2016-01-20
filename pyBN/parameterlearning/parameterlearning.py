"""
***********************
Parameter Learning Code
***********************

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu"""



import pandas as pd
import numpy as np


class ParameterLearner(object):

	def __init__(self, BN=None, data=None, dtype=None):
		"""
		Class for Parameter Learning.

		Attributes
		----------

		BN : a BayesNet object
			The known structure for which we learn params
		data : a pandas dataframe or numpy 3d array
			The data from which we estimate the parameters
		dtype : 'pandas' | 'numpy'
			The form of data

		Methods
		-------

		mle : maximum likelihood estimation algorithm
			Estimate the parameters by counting the fraction
			of each parent-self occurance in the data

		bayes : Bayesian estimation
			Estimate the parameters by assuming a Dirichlet
			prior distribution (uniform or user-specified) with
			equivalent sample size, then computing the posterior
			based on the Multinomial conjugate distribution


		Notes
		-----
		"""
		self.BN = BN

		self.data = data
		if is_instance(data, 'pd.DataFrame'):
			self.dtype = 'pandas'
		elif is_instance(data, 'np.ndarray'):
			self.dtype = 'numpy'
			
		self.dtype = dtype


	