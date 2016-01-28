"""
****************
BN Structure 
Distance Metrics
****************

Code for computing the structural Distance
between 2+ Bayesian networks. Note, this
code compares only structure - i.e. the
conditional (in)dependencies. For a comparison
of conditional probability values of 2+ BNs,
see "parameter_distance.py"

For computing distance metrics that include
both structure AND parameters, see "hybrid_distance.py"


Metrics
-------
*hamming*

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

def hamming(x,y):
	pass