"""
*****************
Min Spanning Tree
*****************

Calcuate Minimum Spanning Tree
for a weighted edge dictionary,
where key = rv, value = another dictionary
where key = child of rv, value = edge weight
between rv -> child edge.

NOTE: This is essentially the same data structure
that networkx uses to store their networks.
"""

import numpy as np

def minimum_spanning_tree(edge_dict):
	"""
	Calcuate Minimum Spanning Tree
	for a weighted edge dictionary,
	where key = rv, value = another dictionary
	where key = child of rv, value = edge weight
	between rv -> child edge.

	Arguments
	---------
	*w_edge_dict* : a dictionary, where
		key = rv, value = another dictionary, where
		key = child node of "rv", value = integer/float
		edge weight between the "rv" -> "child" edge

	Returns
	-------
	*mst_edge_dict* : a non-weighted edge dict
		The minimum spanning tree from w_edge_dict, stored
		as a dictionary where key = rv, value = list of
		rv's children

	Effects
	-------
	None

	Notes
	-----
	test:
		input:
			g = {0:{1:3,3:2,8:4},1:{7:4},
				2:{3:6,7:2,5:1},3:{4:1},4:{8:8},
				5:{6:8},6:{},7:{2:2},8:{}}
		output:
			g = {0:[3,1,8],1:[7],2:[5],3:[4],4:[],
				5:[6],6:[],7:[2],8:[]}
		
	 
	"""
	mst_G = dict([(rv,[]) for rv in range(len(edge_dict))])
	nrv = len(mst_G)
	reached = [0]
	unreached = range(1,nrv)

	for k in xrange(nrv):
		source, sink = None, None
		min_cost = np.inf
		for rn in reached:
			e = edge_dict[rn].items()
			e.sort(key = lambda x : x[1])
			for sn, weight in e:
				if sn in unreached and weight < min_cost:
					min_cost = weight
					source = rn
					sink = sn
					break

		if source is None or sink is None:
			break

		# Add e to the minimum spanning tree
		mst_G[source].append(sink)

		# Mark newly include node as reached
		unreached.remove(sink)
		reached.append(sink)

	return mst_G

























