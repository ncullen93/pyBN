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
from pyBN.classes.factorization import Factorization
from pyBN.utils.graph import *

from copy import deepcopy, copy
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
	_phi = Factorization(bn)

	order = copy(list(bn.nodes()))
	order.remove(target)

	#### EVIDENCE PROCESSING ####
	for E, e in evidence.items():
		_phi -= (E,e)
		order.remove(E)

	#### SUM-PRODUCT ELIMINATE VAR ####
	for var in order:
		_phi /= var

	# multiply phi's together if there is evidence
	final_phi = _phi.consolidate()

	return np.round(final_phi.cpt,4)


def marginal_bp_e(bn, target=None, evidence=None, downward_pass=False):
	"""
	Perform Belief Propagation (Message Passing) over a Clique Tree. This
	is sometimes referred to as the "Junction Tree Algorithm" or
	the "Hugin Algorithm".

	It involves an Upward Pass (see [1] pg. 353) along with
	Downward Pass (Calibration) ([1] pg. 357) if the target involves
	multiple random variables - i.e. is a list

	Steps Involved:
		1. Build a Clique Tree from a Bayesian Network
			a. Moralize the BN
			b. Triangulate the graph
			c. Find maximal cliques and collapse into nodes
			d. Create complete graph and make edge weights = sepset cardinality
			e. Using Max Spanning Tree to create a tree of cliques
		2. Assign each factor to only one clique
		3. Compute the initial potentials of each clique
			- multiply all of the clique's factors together
		4. Perform belief propagation based on message passing protocol.


	Arguments
	---------
	*bn* : a BayesNet object

	Returns
	-------


	Notes
	-----

	"""
	# 1: Moralize the graph
	# 2: Triangluate
	# 3: Build a clique tree using max spanning
	# 4: Propagation of probabilities using message passing

	# creates clique tree and assigns factors, thus satisfying steps 1-3
	ctree = CliqueTree(bn) # might not be initialized?
	#G = ctree.G
	#cliques = copy.copy(ctree.V)

	# select a clique as root where target is in scope of root
	root = ctree.V[0]
	if target is not None:
		for v in ctree.V:
			if target in ctree[v].scope:
				root = v
				break

	
	clique_ordering = ctree.dfs_postorder(root=root)

	# UPWARD PASS
	# send messages up the tree from the leaves to the single root
	for i in clique_ordering:
		#clique = ctree[i]
		for j in ctree.parents(i):
			ctree[i] >> ctree[j] 
			#clique.send_message(ctree[j])
		# if root node, collect its beliefs
		#if len(ctree.parents(i)) == 0:
			#ctree[root].collect_beliefs()
	ctree[root].collect_beliefs()
	marginal_target = ctree[root].marginalize_over(target)

	# DOWNWARD PASS
	if downward_pass == True:
		# send messages down the tree from the root to the leaves
		# (not needed unless *target* involves more than one variable)
		new_ordering = list(reversed(clique_ordering))
		for j in new_ordering:
			for i in ctree.children(j):
				ctree[j] >> ctree[i]
			# if leaf node, collect its beliefs
			if len(ctree.children(j)) == 0:                    
				ctree[j].collect_beliefs()

	return marginal_target

	# beliefs hold the answers









