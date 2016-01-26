


class Factorization(object):
	"""
	Class for Factorization which holds
	a collection of factors as a dict.

	Therefore, creating a Factorization
	requires a dictionary of Factors.
	
	"""


	def __init__(self, factor_dict):
		"""
		Create a Factorization object

		Arguments
		---------
		*factor_dict* : a dictionary
			key = main RV, value = Factor object
		"""
		self.factor_dict = factor_dict

	def num_parents(self, rv):
		return len(self.factor_dict[rv].scope)-1