"""
Topological sort from a Factor dictionary
"""

def topsort(bn):
	"""
	Generator of nodes in topological sort order.
	"""
	queue = [rv for rv in bn.F.keys() if bn.F[rv].num_parents()==0]
	visited = []
	while queue:
		vertex = queue.pop(0)
		if vertex not in visited:
			yield vertex
			visited.append(vertex)
			for node in bn.F.keys():
				if vertex in bn.F[node].parents():
					queue.append(node)