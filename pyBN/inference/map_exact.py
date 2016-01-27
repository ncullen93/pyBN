"""
************************************
Exact Maximum A Posteriori Inference
************************************

Perform exact MAP inference over a BayesNet object,
with or without evidence.

Eventually, there will be a wrapper function "map_exact"
for all of the algorithms, and users can choose their method as
an argument to that function.

Exact MAP Inference Algorithms
------------------------------

	- Max-Sum Variable Elimination


References
----------
[1] Koller, Friedman (2009). "Probabilistic Graphical Models."

"""

__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""

import json
import copy
import numpy as np

from pyBN.classes.factor import Factor


def map_ve_e(bn,
			target=None,
			evidence={}):
	"""
	Perform Max-Sum Variable Elimination over a BayesNet object
	for exact maximum a posteriori inference.

	Parameters
	----------


	Returns
	-------


	Effects
	-------


	Notes
	-----

	"""
	assert (target not in evidence.keys()), 'Target is already in Evidence..'

	temp_F = [Factor(bn,var) for var in bn.nodes()]

	map_list = []

	#### ORDER HANDLING ####
	order = copy.copy(list(bn.nodes()))
	order.remove(target)
	
	##### EVIDENCE #####
	if len(evidence)>0:
		assert isinstance(evidence, dict), 'Evidence must be Dictionary'
		temp=[]
		for rv, val in evidence.items():
			for f in temp_F:
				if len(f.scope)>1 or rv not in f.scope:
					temp.append(f)
				if rv in f.scope:
					f.reduce_factor(rv,val)
			order.remove(rv)
		temp_F=temp
	
	#### ALGORITHM ####
	for var in order:
		relevant_factors = [f for f in temp_F if var in f.scope]
		irrelevant_factors = [f for f in temp_F if var not in f.scope]

		# mutliply all relevant factors
		fmerge = relevant_factors[0]
		for i in range(1,len(relevant_factors)):
			fmerge.multiply_factor(relevant_factors[i])
		
		map_list.append(copy.deepcopy(fmerge))
		fmerge.maxout_var(var)

		irrelevant_factors.append(fmerge) # add sum-prod factor back in
		temp_F = irrelevant_factors

	# Traceback MAP
	assignment={}
	for m in reversed(map_list):
		#var = m.var
		var = list(set(m.scope) - set(assignment.keys()))[0]
		m.reduce_factor_by_list([[k,v] for k,v in assignment.items() \
									if k in m.scope and k!=var])
		assignment[var] = bn.values(var)[(np.argmax(m.cpt) / m.stride[var]) % m.card[var]]
	
	sol = assignment
	sol.update(evidence)
	#print json.dumps(sol,indent=2)
	val=round(temp_F[0].cpt[0],4)
	return sol, val









