"""
**************
Markov Blanket
**************
"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""


def markov_blanket(bn):
	"""
	A markov blanket is a node's parents, children, and 
	its children's parents (i.e. spouses)

	Arguments
	---------
	*bn* : BayesNet object

	Returns
	-------
	*mb* :  a dictionary where each key 
		is a node and the value is a list of
		the key-node's markov blanket
	"""
	mb = dict([(rv,bn.parents(rv)+bn.children(rv)) for rv in bn.nodes()])

	for rv in bn.V:
		for child in bn.children(rv):
			for c_parent in bn.parents(child):
				if c_parent != rv:
					mb[rv].append(c_parent) # add spouse
	return mb

def resolve_markov_blanket(Mb, data):
	"""
	Resolves Markov blanket by returning
	a directed graph - this is actually a PDAG,
	so the edges still need to be oriented by calling
	some version of the "orient_edges" function in
	pyBN.structure_learn.orient_edges module.

	This algorithm is adapted from Margaritis.

	Arguments
	---------
	*Mb* : a dictionary, where
		key = rv and value = list of vars in rv's markov blanket

	*data* : a nested numpy array
		The dataset used to learn the Mb

	Returns
	-------
	*edge_dict* : a dictionary, where
		key = rv and value = list of rv's children

	Effects
	-------
	None

	Notes
	-----
	"""
	edge_dict = dict([(rv,[]) for rv in range(n_rv)])
	for X in range(n_rv):
		for Y in Mb[X]:
			# X and Y are direct neighbors if X and Y are dependent
			# given S for all S in T, where T is the smaller of
			# B(X)-{Y} and B(Y)-{X}
			if len(Mb[X]) < len(Mb[Y]):
				T = copy(Mb[X]) # shallow copy is sufficient
				if Y in T:
					T.remove(Y)
			else:
				T = copy(Mb[Y]) # shallow copy is sufficient
				if X in T:
					T.remove(X)

			# X and Y must be dependent conditioned upon
			# EVERY POSSIBLE COMBINATION of T
			direct_neighbors=True
			for i in range(len(T)):
				for S in itertools.combinations(T,i):
					cols = (X,Y) + tuple(S)
					pval = mi_test(data[:,cols])
					if pval > alpha:
						direct_neighbors=False
			if direct_neighbors:
				if Y not in edge_dict[X] and X not in edge_dict[Y]:
					edge_dict[X].append(Y)
				if X not in edge_dict[Y]:
					edge_dict[Y].append(X)
	return edge_dict







