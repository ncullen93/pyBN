"""
**************
Markov Blanket
**************
"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""


def get_markov_blanket(bn):
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
	mb = dict([(rv,[]) for rv in bn.V])

	for rv in bn.V:
		mb[rv].extend(bn.data[rv]['parents']) # add parents
		mb[rv].extend(bn.data[rv]['children'])
		for child in bn.data[rv]['children']:
			for c_parent in bn.data[child]['parents']:
				if c_parent != rv:
					mb[rv].append(c_parent)
	return mb