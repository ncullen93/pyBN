"""
*****************
Graph Theory Code
*****************

This file mostly contains code for writing
my own implementationsof a few networkx functions
that are commonly used - so that pyBN has
no reliance on networkx.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
from numba import jit, int32

def top_sort(adj_list):
	seen = []
	order = []
	explored = []

	for v in range(len(adj_list)): 
		if v in explored:
			continue
		fringe = [v] 
		while fringe:
			w = fringe[-1]
			if w in explored:
				fringe.pop()
				continue
			seen.append(w)
			new_nodes = []
			for n in adj_list[w]:
				if n not in explored:
					if n in seen:
						print "CYCLE"
						break
					new_nodes.append(n)
			if new_nodes:
				fringe.extend(new_nodes)
			else:   
				explored.append(w)
				order.append(w)
				fringe.pop() 
	return list(reversed(order))











