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
	return sol[target], val

def map_opt_e(bn, evidence={}):
	"""
	Solve MAP Inference as a dynamic programming
	problem, where the solution is built up from
	the leaf nodes by solving subproblems of
	maximal probability rv assignments at each node

	Arguments
	---------
	*bn* : a BayesNet object

	*evidence* : a dictionary, where
		key = rv and value = rv's value

	Returns
	-------
	*sol* : a dictionary, where
		key = rv and value = maximal assignment

	Effects
	-------
	None

	Notes
	-----
	decision variables:
		variable for each set of values in a cpt
	objective:
		minimize sum of negative log probabilities
	constraints:
		- sum for all variables in a cpt must be 1
		- intersection of variables between cpt must agree
	"""
	if evidence:
			assert isinstance(evidence, dict), 'Evidence must be in dictionary form'

	model = LpProblem("MAP Inference",LpMinimize)

	node_var_dict=dict([(rv,[]) for rv in bn.nodes()])
	node_weight_dict = dict([(rv,[]) for rv in bn.nodes()])

	for node_idx,node in enumerate(bn.nodes()):
		for cpt_idx,cpt_val in enumerate(bn.cpt(node)):
			new_var = LpVariable(str(str(node_idx)+'-'+str(cpt_idx)),0,1,LpInteger)
			var_list.append(new_var)
			node_var_dict[node].append(new_var)
			node_weight_dict[node].append(-np.log(cpt_val))
			weight_list.append(-np.log(cpt_val))

	model += np.dot(weight_list,var_list) # minimizes -1*var*probability

	# constraint set 1
	# exactly one choice from each factor
	k = 0
	for rv in bn.nodes():
		cell = node_var_dict[rv]
		model += np.sum(cell) == 1, "Factor Sum Constraint" + str(k)
		k+=1

	# constraint set 2
	# intersection of factors must agree
	


	#add constraint set 3
	#all evidence variables set = 1
	if evidence:
		for k,v in evidence.items():
			ev = str(k)+'='+str(v)
			model += name_var_dict[ev] == 1, 'Evidence Constraint' + str(k)
			k+=1
			
	model.solve()
	max_inference = dict([(v.name,v.varValue) for v in model.variables() if v.varValue == 1])


	

	





