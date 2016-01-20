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

from pyBN.independence.constraint_tests import mi_test

def pc(bn, data, pval=0.05):
	"""
	Path Condition algorithm for structure learning. This is a
	good test, but has some issues with test reliability when
	the size of the dataset is small. The Necessary Path
	Condition (NPC) algorithm can solve these problems.

	Speed Test (mean -> 1000 iterations)
	------------------------------------
	*bnlearn* -> 1.47 milliseconds
		gs(lizards)
	*pyBN* -> 1.79 milliseconds
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
	"""
	rv_card = np.amax(data, axis=0)
	n_rv = len(rv_card)
	#Start with a complete, undirected graph G'
	edge_dict = dict([(i,[j for j in range(n_rv) if i!=j]) for i in range(n_rv)])
	Z_dict = dict([(x,[]) for x in xrange(n_rv)])
	stop = False
	i = 1
	#Repeat:
	while not stop:
		#	For each x \in X:
		for x in xrange(n_rv):
		#		For each y in Adj(x):
			for y in edge_dict[x]:
		#			Test whether there exists some Z in Adj(X)-{Y}
		#			with |Z| = i, such that I(X,Y|Z)
				for z in itertools.combinations(edge_dict[x],i):
					if y not in z:
						cols = (x,y) + z
						pval_xy_z = mi_test(data[:,cols])
			#			If there exists such a set S:
						if (pval_xy_z > pval):
							#print 'Removing edge: ' , x , '-', y
			#				Make Z_xy <- Z
							Z_dict[x].append({y:z})
			#				Remove X-Y link from G'
							edge_dict[x].remove(y)
							edge_dict[y].remove(x)
		#	i <- i + 1
		i += 1
	#Until |Adj(X)| <= i (forall x\inX)
		stop = True
		for x in xrange(n_rv):
			if (len(edge_dict[x]) > i-1):
				stop = False
				break

	bn.set_structure(edge_dict, rv_card)

	return edge_dict






