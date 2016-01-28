"""
****************
BN Parameter 
Distance Metrics
****************

Code for computing the Parametric distance 
between 2+ Bayesian Networks using various
metrics. Parametric distance only involves a
comparison of the conditional probability values.
For structural distance metrics, see "structure_distance.py"

Since a Bayesian network is simply joint probability distribution,
any distibution distance metric can be used.

Metrics
-------
*kl_divergence*
*js_divergence*
*euclidean*
*manhattan*
*mutual_information*
*hellinger*
*minkowski*

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

def kl_divergence(x,y):
	pass