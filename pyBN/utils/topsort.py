"""
Topological sort from a Factor dictionary
"""

def topsort(E):
	"""
	List of nodes in topological sort order from edge dict
	where key = rv and value = list of rv's children
	"""
	queue = []
	for rv in E.keys():
		prior=True
		for p in E.keys():
			if rv in E[p]:
				prior=False
		if prior==True:
			queue.append(rv)
	
	visited = []
	while queue:
		vertex = queue.pop(0)
		if vertex not in visited:
			visited.append(vertex)
			queue.extend(E[vertex]) # add all vertex's children
	return visited