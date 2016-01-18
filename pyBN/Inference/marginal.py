"""
******************
Marginal Inference
******************

Perform marginal inference over a BayesNet object.

We support exact/approx inference, with or without evidence.

Exact Marginal Inference Algorithms:
	
	- Sum-Product Variable Elimination
	- Clique Tree Message Passing (Belief Propagation)

Approximate Marginal Inference Algorithms:
	
	- Forward Sampling
	- Likelihood Weighted Sampling
	- Gibbs (MCMC) Sampling
	- Loopy Belief Propagation

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


def forward_sample(self, n=1000):
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
	sampler = Sampler(self.BN)
	sampler.forward_sample(n=n)


def gibbs_sample(self, n=1000):
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
	sampler=Sampler(self.BN)
	sampler.gibbs_sample(n=n)
	

def lw_sample(self, n=1000, evidence={}):
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
	sampler=Sampler(self.BN)
	sampler.likelihood_weighted_sample(target,evidence,n)



