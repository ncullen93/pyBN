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