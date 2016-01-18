"""
************************************
Exact Maximum A Posteriori Inference
************************************

Perform exact MAP inference over a BayesNet object,
with or without evidence.

Exact MAP Inference Algorithms
------------------------------

	- Max-Sum Variable Elimination


Notes
-----
All above functions are implemented but not tested/documented.

"""

__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""


def max_sum_ve(self, evidence=None, order=None):
	"""
	Overview
	--------


	Parameters
	----------


	Returns
	-------


	Notes
	-----

	"""
	factorization = Factorization(self.bn)
	factorization.variable_elimination(marginal=False)