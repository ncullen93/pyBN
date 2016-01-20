"""
******************
Orient Edges from
Structure Learning
******************

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

def orient_edges(edges, witnesses):
	"""
	Orient edges based on the well-established
	rules in the literature.

	See Koller pg. 89

	Arguments
	---------
	*edge_dict* : a dictionary
		Dictionary of undirected edges, so 
		there are duplicates

	*z_dict* : a dictionary

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
	n_rv=3

	dedges = [x[:] for x in edges]
	for edge in dedges:
		edge.append('u')

	for edge1 in reversed(dedges):
		for edge2 in reversed(dedges):
			if (edge1 in dedges) and (edge2 in dedges):
				if edge1[0] == edge2[1]:
					if (([edge1[1], edge2[0], [edge1[0]]] not in witnesses) and ([edge2[0], edge1[1], [edge1[0]]] not in witnesses)): 
						dedges.append([edge1[1], edge1[0]])
						dedges.append([edge2[0], edge2[1]])
						dedges.remove(edge1)
						dedges.remove(edge2)
				elif edge1[1] == edge2[0]:
					if (([edge1[0], edge2[1], [edge1[1]]] not in witnesses) and ([edge2[1], edge1[0], [edge1[1]]] not in witnesses)): 
						dedges.append([edge1[0], edge1[1]])
						dedges.append([edge2[1], edge2[0]])
						dedges.remove(edge1)
						dedges.remove(edge2)
				elif edge1[1] == edge2[1] and edge1[0] != edge2[0]:
					if (([edge1[0], edge2[0], [edge1[1]]] not in witnesses) and ([edge2[0], edge1[0], [edge1[1]]] not in witnesses)): 
						dedges.append([edge1[0], edge1[1]])
						dedges.append([edge2[0], edge2[1]])
						dedges.remove(edge1)
						dedges.remove(edge2)
				elif edge1[0] == edge2[0] and edge1[1] != edge2[1]:
					if (([edge1[1], edge2[1], [edge1[0]]] not in witnesses) and ([edge2[1], edge1[1], [edge1[0]]] not in witnesses)): 
						dedges.append([edge1[1], edge1[0]])
						dedges.append([edge2[1], edge2[0]])
						dedges.remove(edge1)
						dedges.remove(edge2)

	# define helper method "exists_undirected_edge"
	def exists_undirected_edge(one_end, the_other_end):
		for edge in dedges:
			if len(edge) == 3:
				if (edge[0] == one_end and edge[1] == the_other_end):
					return True
				elif (edge[1] == one_end and edge[0] == the_other_end):
					return True
		return False
	# use right hand rules to improve graph until convergence (Koller 89)
	olddedges = []
	while (olddedges != dedges):
		olddedges = [x[:] for x in dedges]
		for edge1 in reversed(dedges):
			for edge2 in reversed(dedges):

				# rule 1
				inverted = False
				check1, check2 = False, True
				if (edge1[1] == edge2[0] and len(edge1) == 2 and len(edge2) == 3):
					check1 = True
				elif (edge1[1] == edge2[1] and len(edge1) == 2 and len(edge2) == 3):
					check = True
					inverted = True 
				for edge3 in dedges:
					if edge3 != edge1 and ((edge3[0] == edge1[0] and edge3[1]
						== edge2[1]) or (edge3[1] == edge1[0] and edge3[0]
						== edge2[1])):
						check2 = False
				if check1 == True and check2 == True:
					if inverted:
						dedges.append([edge1[1], edge2[0]])
					else:
						dedges.append([edge1[1], edge2[1]])
					dedges.remove(edge2)

				# rule 2
				check1, check2 = False, False
				if (edge1[1] == edge2[0] and len(edge1) == 2 and len(edge2) == 2):
					check1 = True
				for edge3 in dedges:
					if ((edge3[0] == edge1[0] and edge3[1]
						== edge2[1]) or (edge3[1] == edge1[0] and edge3[0]
						== edge2[1]) and len(edge3) == 3):
						check2 = True
				if check1 == True and check2 == True:
					if edge3[0] == edge1[0]:
						dedges.append([edge3[0], edge3[1]])
					elif edge3[1] == edge1[0]:
						dedges.append([edge3[1], edge3[0]])
					dedges.remove(edge3)

				# rule 3
				check1, check2 = False, False
				if len(edge1) == 2 and len(edge2) == 2:
					if (edge1[1] == edge2[1] and edge1[0] != edge2[0]):
						check1 = True
				for v in range(n_rv):
					if (exists_undirected_edge(v, edge1[0]) and
						exists_undirected_edge(v, edge1[1]) and
						exists_undirected_edge(v, edge2[0])):
						check2 = True
						if check1 == True and check2 == True:
							dedges.append([v, edge1[1]])
							for edge3 in dedges:
								if (len(edge3) == 3 and ((edge3[0] == v and edge3[1]
									== edge1[1]) or (edge3[1] == v and edge3[0] ==
									edge1[1]))):
									dedges.remove(edge3)


	# return one possible graph skeleton from the pdag class found
	for x in range(len(dedges)):
		if len(dedges[x]) == 3:
			dedges[x] = dedges[x][:2]
	#print dedges
	return dedges
