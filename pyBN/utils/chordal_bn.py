"""
Algorithm for making
a Bayesian network chordal -
i.e. no min cycles of more than
three random variables.

"""
from copy import copy
import networkx as nx

def make_chordal(bn, v=None,e=None):
	"""
	This function creates a chordal graph - i.e. one in which there
	are no cycles with more than three nodes.

	Can supply a v_list and e_list for chordal graph of any random graph..

	We start from the moral graph, so if that is already chordal then it
	will return that.

	Algorithm from Cano & Moral 1990 ->
	'Heuristic Algorithms for the Triangulation of Graphs'


	Parameters
	----------
	*v* : a list (optional)
	    A vertex list

	*e* : a list (optional)
	    An edge list


	Returns
	-------
	*G* : a network Digraph object

	Effects
	-------
	None

	Notes
	-----
	Where is this used? Do we need to use networkx?       


	"""
	chordal_E = bn.moralized_edges() # start with moral graph

	# if moral graph is already chordal, no need to alter it
	if not is_chordal(chordal_E):            
		temp_E = copy.copy(chordal_E)
		temp_V = []

		# if v and e is supplied, skip all the rest
		if v and e:
			chordal_E = copy.copy(e)
			temp_E = copy.copy(chordal_E)
			temp_V = copy.copy(v)
		else:
			temp_G = nx.Graph()
			temp_G.add_edges_from(chordal_E)
			degree_dict = temp_G.degree()
			temp_V = sorted(degree_dict, key=degree_dict.get)
		#print temp_V
		for v in temp_V:
			#Add links between the pairs nodes adjacent to Node i
			#Add those links to chordal_E and temp_E
			adj_v = set([n for e in temp_E for n in e if v in e and n!=v])
			for a1 in adj_v:
				for a2 in adj_v:
					if a1!=a2:
						if [a1,a2] not in chordal_E and [a2,a1] not in chordal_E:
							chordal_E.append([a1,a2])
							temp_E.append([a1,a2])
			# remove Node i from temp_V and all its links from temp_E 
			temp_E2 = []
			for edge in temp_E:
				if v not in edge:
					temp_E2.append(edge)
			temp_E = temp_E2

	
	G = nx.Graph()
	G.add_edges_from(chordal_E)
	return G


def is_chordal(edge_list):
	"""
	Check if the graph is chordal/triangulated.

	Parameters
	----------
	*edge_list* : a list of lists (optional)
	The edges to check (if not self.E)

	Returns
	-------
	*nx.is_chordal(G)* : a boolean
	Whether the graph is chordal or not

	Effects
	-------
	None

	Notes
	-----
	Again, do we need networkx for this? Eventually should
	write this check on our own.

	"""
	G = nx.Graph(list(edge_list))
	return nx.is_chordal(G)