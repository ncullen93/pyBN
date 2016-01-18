"""
************************
Exact Marginal Inference
************************

Perform marginal inference over a BayesNet object. We support 
exact/approx inference, with or without evidence.

Exact Marginal Inference Algorithms
-----------------------------------
	
	- Sum-Product Variable Elimination
	- Clique Tree Message Passing (Belief Propagation)


Notes
-----
All above functions are here, but they are not tested/documented.

"""

__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""



def sum_product_ve(bn, target=None, evidence=None, order=None):
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
	factorization = Factorization(bn)
	factorization.marginal_ve(target,evidence,order)


def clique_tree_bp(bn, target=None, evidence=None, downward_pass=True):
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
	ctree = CliqueTree(bn)
	ctree.message_passing(target, evidence, downward_pass)






