"""
******************************
Approximate Marginal Inference
******************************

Perform approx. marginal inference over a BayesNet object,
with or without evidence.

Approximate Marginal Inference Algorithms
-----------------------------------------
	
	- Forward Sampling
	- Likelihood Weighted Sampling
	- Gibbs (MCMC) Sampling
	- Loopy Belief Propagation

Notes
-----
All above functions are here, but they are not tested/documented.

"""

__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""




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


def loopy_bp(self, target=None, evidence=None):
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
	cgraph = ClusterGraph(self.BN)
	cgraph.loopy_belief_propagation(target, evidence)