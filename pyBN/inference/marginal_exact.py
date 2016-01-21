"""
************************
Exact Marginal Inference
************************

Perform exact marginal inference over a BayesNet object,
with or without evidence.

Eventually, there will be a wrapper function "marginal_exact"
for all of the algorithms, and users can choose their method as
an argument to that function.

Exact Marginal Inference Algorithms
-----------------------------------
	
	- Sum-Product Variable Elimination
	- Clique Tree Message Passing (Belief Propagation)


References
----------
[1] Koller, Friedman (2009). "Probabilistic Graphical Models."

"""

__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""



def marginal_var_elim(bn, 
					target=None, 
					evidence=None, 
					order=None):
	"""
	Perform Sum-Product Variable Elimination on
	a Discrete Bayesian Network

	Parameters
	----------
	bn=BayesNet
	target=[], 
	evidence={}, 
	order=[]

	Returns
	-------


	Notes
	-----
	This function is fully implemented, but not tested

	"""
	self.temp_f_list = [Factor(self.BN,var) for var in self.BN.V]
	map_list = []

	#### ORDER HANDLING ####
	if not order:
		order = copy.copy(self.BN.V)
		if isinstance(target,list):
			for t in target:
				order.remove(t)
		else: 
			order.remove(target)
	if isinstance(target,list):
		for t in target:
			if target in order:
				order.remove(t)
	else:
		if target in order:
			order.remove(target)

	##### EVIDENCE #####

	if len(evidence)>0:
		assert isinstance(evidence, dict), 'Evidence must be Dictionary'
		temp=[]
		for obs in evidence.items():
			for f in self.temp_f_list:
				if len(f.scope)>1 or obs[0] not in f.scope:
					temp.append(f)
				if obs[0] in f.scope:
					f.reduce_factor(obs[0],obs[1])
			order.remove(obs[0])
		self.temp_f_list=temp

	#### ALGORITHM ####
	for var in order:
		relevant_factors = [f for f in self.temp_f_list if var in f.scope]
		irrelevant_factors = [f for f in self.temp_f_list if var not in f.scope]

		# mutliply all relevant factors
		fmerge = relevant_factors[0]
		for i in range(1,len(relevant_factors)):
			fmerge.multiply_factor(relevant_factors[i])
		## only difference between marginal and map
		if marginal==False:
			map_list.append(copy.deepcopy(fmerge))
			fmerge.maxout_var(var)
		else:
			fmerge.sumout_var(var) # remove var from factor

		irrelevant_factors.append(fmerge) # add sum-prod factor back in
		self.temp_f_list = irrelevant_factors

	
	
	marginal = self.temp_f_list[0]
	# multiply final factors in factor_list
	if len(self.temp_f_list) > 1:
		for i in range(1,len(self.temp_f_list)):
			marginal.multiply_factor(self.temp_f_list[i])
	marginal.normalize()

	self.sol=marginal.cpt


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






