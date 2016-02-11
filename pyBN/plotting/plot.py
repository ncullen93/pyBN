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
import matplotlib.image as mpimg
import pylab
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import subprocess
import sys
from PIL import Image
		

def plot_nx(bn,**kwargs):
	"""
	Draw BayesNet object from networkx engine
	"""
	g = nx.DiGraph(bn.E)
	pos = graphviz_layout(g,'dot')
	#node_size=600,node_color='w',with_labels=False
	nx.draw_networkx(g,pos=pos, **kwargs)
	plt.axis('off')
	plt.show()

def plot_inline(bn, h=350, w=450):
	def execute(command):
		command = ' '.join(command)
		process = subprocess.Popen(command, 
			shell=True, 
			stdout=subprocess.PIPE, 
			stderr=subprocess.STDOUT)

		output = process.communicate()[0]
		returncode = process.returncode

		if returncode == 0:
			return True, output
		else:
			return False, output

	cmd = ['mkdir' , 'pyBN/drawing/images']
	p = execute(cmd)

	G = nx.DiGraph(bn.E)
	write_dot(G,"pyBN/drawing/images/bn.dot")

	cmd = ['/usr/local/bin/dot', '-Tpng' , 
	'pyBN/drawing/images/bn.dot', 
	'>pyBN/drawing/images/bn.png']
	p = execute(cmd)
	
	im = Image.open("pyBN/drawing/images/bn.png")
	out = im.resize((w,h))

	cmd = ['rm' , '-r', 'pyBN/drawing/images']
	p = execute(cmd)

	return out

def plot_gv(bn):
	def execute(command):
		command = ' '.join(command)
		process = subprocess.Popen(command, 
			shell=True, 
			stdout=subprocess.PIPE, 
			stderr=subprocess.STDOUT)

		output = process.communicate()[0]
		returncode = process.returncode

		if returncode == 0:
			return True, output
		else:
			return False, output

	cmd = ['mkdir' , 'pyBN/drawing/images']
	p = execute(cmd)

	G = nx.DiGraph(bn.E)
	write_dot(G,"pyBN/drawing/images/bn.dot")

	cmd = ['/usr/local/bin/dot', '-Tpng' , 
	'pyBN/drawing/images/bn.dot', 
	'>pyBN/drawing/images/bn.png']

	p = execute(cmd)
	plt.figure(facecolor="white")
	img=mpimg.imread('pyBN/drawing/images/bn.png')
	_img = plt.imshow(img, aspect='auto')
	plt.axis('off')
	plt.show(_img)

	cmd = ['rm' , '-r', 'pyBN/drawing/images']
	p = execute(cmd)











		
