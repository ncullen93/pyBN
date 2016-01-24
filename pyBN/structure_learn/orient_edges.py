"""
******************
Orient Edges from
Structure Learning
******************

[1] Chickering. "Learning Equivalence Classes of Bayesian
Network Structues"
http://www.jmlr.org/papers/volume2/chickering02a/chickering02a.pdf
"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.independence.constraint_tests import mi_test

import itertools
from copy import copy

def orient_edges_gs(edge_dict, B, data, alpha):
	"""
	Orient edges based on the rules presented
	in Margaritis' Thesis pg. 35

	# if there exists a variable Z in N(X)-N(Y)-{Y}
	# such that Y and Z are dependent given S+{X} for
	# all S subset of T, where
	# T is smaller of B(Y)-{X,Z} and B(Z)-{X,Y}

	Arguments
	---------
	*edge_dict* : a dictionary, where
		key = node and value = list
		of neighbors for key. Note: there
		MUST BE duplicates in edge_dict ->
		i.e. each edge should be in edge_dict
		twice since Y in edge_dict[X] and
		X in edge_dict[Y]

	*blanket_dict* : a dictionary, where
		key = node and value = list of
		nodes in the markov blanket of node

	Returns
	-------
	*d_edge_dict* : a dictionary
		Dictionary of directed edges, so
		there are no duplicates

	Effects
	-------
	None

	Notes
	-----

	"""
	for X in edge_dict.keys():
		for Y in edge_dict[X]:
			nxy = set(edge_dict[X]) - set(edge_dict[Y]) - {Y}

			for Z in nxy:
				by = set(B[Y]) - {X} - {Z}
				bz = set(B[Z]) - {X} - {Y}
				T = min(by,bz)
				if len(T)>0:
					for i in range(len(T)):
						for S in itertools.combinations(T,i):
							cols = (Y,Z,X) + tuple(S)
							pval = mi_test(data[:,cols])
							if pval < alpha:
								edge_dict[X].remove(Y)
							else:
								edge_dict[Y].remove(X)
				else:
					cols = (Y,Z,X)
					pval = mi_test(data[:,cols])
					if pval < alpha:
						edge_dict[X].remove(Y)
					else:
						edge_dict[Y].remove(X)
	return edge_dict
















