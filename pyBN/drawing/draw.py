"""
*******
Drawing 
*******

Code for drawing BayesNet objects, built on
the graphviz framework.
"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import networkx as nx
import pydot
import graphviz as gv
from graphviz import dot
import matplotlib.pyplot as plt
import pylab

def draw_bn(bn, method='nx',save=False, **kwargs):
	"""
	Draw a BayesNet object, using graphviz engine
	or networkx engine.

	Arguments
	---------
	*bn* : a BayesNet object

	*save* : a boolean
		Whether to save the plot to file

	*kwargs* : comma-separated Arguments
		Additional plotting parameters

	Returns
	-------
	None

	Effects
	-------
	- Saves to file if specified

	Notes
	-----
	"""
	if method == 'nx':
		draw_nx(bn,**kwargs)

	elif method == 'gv':
		nodes = list(bn.nodes()) # good
		edges = list(bn.edges())
		styles = {
					'graph': {
							'label': 'A Fancy Graph',
							'fontsize': '16',
							'fontcolor': 'black',
							'bgcolor': 'white',
							'rankdir': 'BT',
					},
					'nodes': {
							'fontname': 'Helvetica',
							'shape': 'cicle',
							'fontcolor': 'black',
							'color': 'black',
							#'style': 'filled',
							#'fillcolor': '#006699',
					},
					'edges': {
							'style': 'solid',
							'color': 'black',
							'arrowhead': 'open',
							'fontname': 'Courier',
							'fontsize': '12',
							'fontcolor': 'black',
					}
				}

		draw_gv(nodes,edges,styles)

def draw_nx(bn,**kwargs):
	"""
	Draw BayesNet object from networkx engine
	"""
	g = bn.get_networkx()
	pos = nx.graphviz_layout(g,'dot')
	#node_size=600,node_color='w',with_labels=False
	nx.draw_networkx(g,pos=pos, **kwargs)
	plt.axis('off')
	plt.show()

def draw_gv(nodes,
			edges,
			styles):
	"""
	Draw BayesNet object from graphviz engine.
	"""
	def add_nodes(graph, nodes):
		"""
		Add nodes to a graphviz graph

		Arguments
		---------
		*graph* : a graphviz graph

		*nodes* : a list of strings or tuples, where
			for each tuple element, the first
			element is the node and the second
			element is a dictionary of attributes.
			Ex: nodes = ['A', 'B', ('C', {})]

		Returns
		-------
		*graph* : a modified graphviz graph

		"""
		for n in nodes:
			if isinstance(n, tuple):
				graph.node(n[0], **n[1])
			else:
				graph.node(n)
		return graph

	def add_edges(graph, edges):
		"""
		Add edges to a graphviz graph

		Arguments
		---------
		*graph* : a graphviz graph

		*edges* : a list of tuples, where
			for each tuple element, the first
			two elements are the nodes and the
			third element is a dictionary of
			attributes.
			Ex: edges = [
					('A', 'B'),
					('B', 'C'),
					(('A', 'C'), {}),
				]
		Returns
		-------
		*graph* : a modified graphviz graph

		"""
		for e in edges:
			if isinstance(e[0], tuple):
				graph.edge(*e[0], **e[1])
			else:
				graph.edge(*e)
		return graph

	def apply_styles(graph, styles):
		"""
		Apply styles to a graphviz graph.

		Arguments
		---------
		*graph* : a graphviz object

		*styles* : a dictionary of dictionaries
			Ex:
				styles = {
					'graph': {
							'label': 'A Fancy Graph',
							'fontsize': '16',
							'fontcolor': 'white',
							'bgcolor': '#333333',
							'rankdir': 'BT',
					},
					'nodes': {
							'fontname': 'Helvetica',
							'shape': 'hexagon',
							'fontcolor': 'white',
							'color': 'white',
							'style': 'filled',
							'fillcolor': '#006699',
					},
					'edges': {
							'style': 'dashed',
							'color': 'white',
							'arrowhead': 'open',
							'fontname': 'Courier',
							'fontsize': '12',
							'fontcolor': 'white',
					}
				}
		"""
		graph.graph_attr.update(
			('graph' in styles and styles['graph']) or {}
		)
		graph.node_attr.update(
			('nodes' in styles and styles['nodes']) or {}
		)
		graph.edge_attr.update(
			('edges' in styles and styles['edges']) or {}
		)
		return graph

	g = gv.Digraph(format='png')
	g = add_nodes(g, nodes)
	g = add_edges(g, edges)
	g = apply_styles(g, styles)
	g.render('img/g')











		
