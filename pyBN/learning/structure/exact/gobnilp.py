"""
GOBNILP Solver for Exact Bayesian network
Structure Learning with Integer Linear
Optimization.

This code relies on the "pyGOBN" package, which
implements a Python wrapper for the GOBNILP
project - which itself builds on the SCIP
Optimization engine for building Integer Linear
Programs to solve the BN structure learning problem
EXACTLY.

Th pyGOBN code - linked in this file - allows the user
to create or alter a vast number of global parameter settings 
(i.e. wall-time, parent limits, sparsity, equivalent 
sample size, etc), along with various network constraints 
(i.e. required edges, disallowed edges, and independence
requirements).

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

def learn_gobnilp(data, settings=None, edge_reqs=None, nonedge_reqs=None, ind_reqs=None):
	#try:
	#	from pyGOBN import *
	#except ImportError:
	#	print 'Failed to import pyGOBN.'

	gobn = GOBN()
	gobn.set_settings(settings)
	gobn.set_constraints(edge_reqs=edge_reqs, nonedge_reqs=nonedge_reqs, ind_reqs=ind_reqs)
