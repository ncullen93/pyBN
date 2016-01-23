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

def orient_edges(edge_dict, blanket_dict):
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
		ARE duplicates in edge_dict.

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
			# orient it Y->X if
			# there exists a variable Z in N(X)-N(Y)-{Y}
			# such that Y and Z are dependent given S+{X} for
			# all S subset of T, where
			# T is smaller of B(Y)-{X,Z} and B(Z)-{X,Y}
			nxy = set(edge_dict[X]) - set(edge_dict[Y]) - set(Y)

			for Z in nxy:
				T = 

				cols = (Y,Z) + tuple(S)+X
				pval = mi_test(data[:,cols])
				if pval < alpha:














