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



def marginal_ve_e(bn, 
				target=None, 
				evidence=None, 
				order=None):
	"""
	Perform Sum-Product Variable Elimination on
	a Discrete Bayesian Network.

	Great candidate for Numba JIT

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
	temp_f_list = [Factor(bn,var) for var in bn.V]
	map_list = []

	#### ORDER HANDLING ####
	if not order:
		order = copy.copy(bn.V)
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
			for f in temp_f_list:
				if len(f.scope)>1 or obs[0] not in f.scope:
					temp.append(f)
				if obs[0] in f.scope:
					f.reduce_factor(obs[0],obs[1])
			order.remove(obs[0])
		temp_f_list=temp

	#### ALGORITHM ####
	for var in order:
		relevant_factors = [f for f in temp_f_list if var in f.scope]
		irrelevant_factors = [f for f in temp_f_list if var not in f.scope]

		# mutliply all relevant factors
		fmerge = relevant_factors[0]
		for i in range(1,len(relevant_factors)):
			fmerge.multiply_factor(relevant_factors[i])
		
		fmerge.sumout_var(var) # remove var from factor

		irrelevant_factors.append(fmerge) # add sum-prod factor back in
		temp_f_list = irrelevant_factors

	
	
	marginal = temp_f_list[0]
	# multiply final factors in factor_list
	if len(temp_f_list) > 1:
		for i in range(1,len(temp_f_list)):
			marginal.multiply_factor(temp_f_list[i])
	marginal.normalize()

	return marginal.cpt


def marginal_bp_e(bn, 
					target=None, 
					evidence=None, 
					downward_pass=True):
	"""
	Perform Message Passing (Belief Propagation) over a clique tree.

	It is Upward Pass as shown in Koller p.353 along with
	Downward Pass (Calibration) from Koller p.357 if target is list


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









