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

from pyBN.classes.factor import Factor
import copy
import numpy as np
import json

def marginal_ve_e(bn, target, evidence={}):
	"""
	Perform Sum-Product Variable Elimination on
	a Discrete Bayesian Network.

	Arguments
	---------
	*bn* : a BayesNet object

	*target* : a list of target RVs

	*evidence* : a dictionary, where
		key = rv and value = rv value

	Returns
	-------
	*marginal_dict* : a dictionary, where
		key = an rv in target and value =
		a numpy array containing the key's
		marginal conditional probability distribution.

	Notes
	-----
	- Mutliple pieces of evidence often returns "nan"...numbers too small?
		- dividing by zero -> perturb values in Factor class
	"""
	temp_F = [Factor(bn,var) for var in bn.nodes()] # topsort order

	#### ORDER HANDLING ####
	order = copy.copy(list(bn.nodes()))
	order.remove(target)

	##### EVIDENCE #####

	if len(evidence)>0:
		assert isinstance(evidence, dict), 'Evidence must be Dictionary'
		temp=[]
		for obs in evidence.items():
			for f in temp_F:
				if len(f.scope)>1 or obs[0] not in f.scope:
					temp.append(f)
				if obs[0] in f.scope:
					f.reduce_factor(obs[0],obs[1])
			order.remove(obs[0])
		temp_F=temp

	#### ALGORITHM ####
	for var in order:
		relevant_factors = [f for f in temp_F if var in f.scope]
		irrelevant_factors = [f for f in temp_F if var not in f.scope]

		# mutliply all relevant factors
		fmerge = relevant_factors[0]
		for i in range(1,len(relevant_factors)):
			fmerge.multiply_factor(relevant_factors[i])
		
		fmerge.sumout_var(var) # remove var from factor

		irrelevant_factors.append(fmerge) # add sum-prod factor back in
		temp_F = irrelevant_factors

	marginal = temp_F[0]
	# multiply final factors in factor_list
	if len(temp_F) > 1:
		for i in range(1,len(temp_F)):
			marginal.multiply_factor(temp_F[i])
	marginal.normalize()

	return marginal.cpt


def marginal_bp_e(bn, target=None, evidence=None, downward_pass=True):
	"""
	Perform Message Passing (Belief Propagation) over a Clique Tree.

	It involves an Upward Pass (see [1] pg. 353) along with
	Downward Pass (Calibration) ([1] pg. 357) if the target involves
	multiple random variables - i.e. is a list


	Parameters
	----------


	Returns
	-------


	Notes
	-----
	- Copied directly from CliqueTree class... not tested.

	"""
	# 1: Moralize the graph
	# 2: Triangluate
	# 3: Build a clique tree using max spanning
	# 4: Propagation of probabilities using message passing

	# creates clique tree and assigns factors, thus satisfying steps 1-3
	ctree = CliqueTree(bn) # might not be initialized?
	G = ctree.G # ugh
	#cliques = copy.copy(ctree.V)

	# select a clique as root where target is in scope of root
	root=np.random.randint(0,len(ctree.V))
	if target:
		root = [node for node in G.nodes() if target in ctree.V[node].scope][0]

	tree_graph = nx.dfs_tree(G,root)
	clique_ordering = list(nx.dfs_postorder_nodes(tree_graph,root))

	# SEND MESSAGES UP THE TREE FROM THE LEAVES TO THE SINGLE ROOT
	for i in clique_ordering:
		clique = ctree.V[i]
		for j in tree_graph.predecessors(i):
			clique.send_message(ctree.V[j])
		# if root node, collect its beliefs
		if len(tree_graph.predecessors(i)) == 0:
			ctree.V[root].collect_beliefs()

	if downward_pass:
		# if target is a list, run downward pass
		new_ordering = list(reversed(clique_ordering))
		for j in new_ordering:
			clique = ctree.V[j]
			for i in tree_graph.successors(j):
				clique.send_message(ctree.V[i])
			# if leaf node, collect its beliefs
			if len(tree_graph.successors(j)) == 0:                    
				ctree.V[j].collect_beliefs()

	return ctree.beliefs

	# beliefs hold the answers









