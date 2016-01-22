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

def orient_edges(edge_dict, block_list):
	"""
	Orient edges based on the well-established
	rules in the literature.

	See Koller pg. 89

	Step 1:
	For every triple of nodes X,Y,Z such that there is an edge
	from X - Z and Z - Y, if Y is NOT the third index in blocklist[X,Y],
	then add edge X->Z , Y->Z.

	Arguments
	---------
	*edge_dict* : a dictionary
		Dictionary of undirected edges, so 
		there are duplicates

	*block_list* : a list of nested 3-tuples
		Each tuples represents a structure such that
		tuple[0] is independent of tuple[1] given tuple[2],
		where tuple[2] is another tuple.
		Clearly, then, there cannot be a V-structure between
		these three+ nodes.

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
	pass
	#{0:[1,2],1:[0],2:[0]}
	#[[0,1],[0,2]]
	#{0:[],1:{2:(0,)}, 2:[]}
	# STEP 1 -> check for V-structures
	# see Margaritis pg 22












