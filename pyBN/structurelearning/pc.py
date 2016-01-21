"""
************
PC Algorithm
************

Pseudo-Code
-----------
Start with a complete, undirected graph G'
i <- 0
Repeat:
	For each x \in X:
		For each y in Adj(x):

			Test whether there exists some S in Adj(X)-{Y}
			with |S| = i, such that I(X,Y|S)

			If there exists such a set S:
				Make S_xy <- S
				Remove X-Y link from G'
	i <- i + 1
Until |Adj(X)| <= i (forall x\inX)

for (each uncoupled meeting X-Z-Y)
	if (Z not in S_xy)
		orient X - Z - Y as X -> Z <- Y
while (more edges can be oriented)
	for (each uncoupled meeting X -> Z - Y)
		orient Z - Y as Z -> Y
	for (each X - Y such that there is a path from X to Y)
		orient X - Y as X -> Y
	for (each uncoupled meeting X - Z - Y such that
		X -> W, Y -> W, and Z - W)
		orient Z - W as Z -> W


References
----------
[1] Abellan, Gomez-Olmedo, Moral. "Some Variations on the
PC Algorithm." http://www.utia.cas.cz/files/mtr/pgm06/41_paper.pdf
[2] Spirtes, Glymour, Scheines (1993) "Causation, Prediction,
and Search."
"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import itertools
import numpy as np

from pyBN.independence.constraint_tests import mi_test_conditional, mi_test_marginal
from pyBN.classes import BayesNet
from pyBN.structurelearning.orient_edges import orient_edges

def pc(data, pval=0.05):
	"""
	Path Condition algorithm for structure learning. This is a
	good test, but has some issues with test reliability when
	the size of the dataset is small. The Necessary Path
	Condition (NPC) algorithm can solve these problems.

	Speed Test (mean -> 1000 iterations)
	------------------------------------
	*bnlearn* -> 1.65 milliseconds
		cextend(gs(lizards))
	*pyBN* -> 1.87 milliseconds
		pc(lizards)

	Arguments
	---------
	*bn* : a BayesNet object
		The object we wish to modify. This can be a competely
		empty BayesNet object, in which case the structure info
		will be set. This can be a BayesNet object with already
		initialized structure/params, in which case the structure
		will be overwritten and the parameters will be cleared.

	*data* : a nested numpy array
		The data from which we will learn -> will code for
		pandas dataframe after numpy works

	Returns
	-------
	*bn* : a BayesNet object
		The network created from the learning procedure, with
		the nodes/edges initialized/changed

	Effects
	-------
	None

	Notes
	-----
	- Z_dict is used to orient the edges -> not implemented yet.
	- Because edge_dict includes double the number of edges right now,
	the number of conditional independence tests this function runs is also
	double the sufficient amount... fix this for better speed.
	"""
	##### FIND EDGES #####
	rv_card = np.amax(data, axis=0)
	n_rv = len(rv_card)
	
	edge_dict = dict([(i,[j for j in range(n_rv) if i!=j]) for i in range(n_rv)])
	block_dict = dict.fromkeys(range(n_rv),[])
	stop = False
	i = 1
	while not stop:
		for x in xrange(n_rv):
			for y in edge_dict[x]:
				if i == 0:
					pval_xy_z = mi_test_marginal(data[:,(x,y)])
					if pval_xy_z > pval:
						edge_dict[x].remove(y)
						edge_dict[y].remove(x)
				else:
					for z in itertools.combinations(edge_dict[x],i):
						if y not in z:
							cols = (x,y) + z
							pval_xy_z = mi_test_conditional(data[:,cols])
							if pval_xy_z > pval:
								block_dict[x] = {y:z}
								edge_dict[x].remove(y)
								edge_dict[y].remove(x)
		i += 1
		stop = True
		for x in xrange(n_rv):
			if (len(edge_dict[x]) > i-1):
				stop = False
				break
	print edge_dict
	print block_dict
	##### ORIENT EDGES #####
	#d_edge_dict = orient_edges(edge_dict, block_list)

	# wont work correctly until edge orientation is figured out
	card_dict = dict(zip(range(len(rv_card)),rv_card))
	bn=BayesNet()
	bn.set_structure(edge_dict, card_dict)
	return bn






