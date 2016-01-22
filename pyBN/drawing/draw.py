"""
*******
Drawing 
*******

Code for drawing BayesNet objects, built on
the graphviz framework.
"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import networkx as nx
import matplotlib.pyplot as plt

def draw_bn(bn, save=False, **kwargs):
	"""
	Draw a BayesNet object, using graphviz engine
	accessed through networkx.

	Arguments
	---------
	*bn* : a BayesNet object

	*save* : a boolean
		Whether to save the plot to file

	Returns
	-------
	None

	Effects
	-------
	- Saves to file if specified

	Notes
	-----
	"""
	g = bn.get_networkx()
	pos = nx.graphviz_layout(g,'dot')
	#node_size=600,node_color='w',with_labels=False
	nx.draw_networkx(g,pos=pos, **kwargs)
	plt.axis('off')
	plt.show()
		
