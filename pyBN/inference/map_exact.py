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

    _phi = [Factor(bn,var) for var in bn.nodes()]

    #### ORDER HANDLING ####
    order = copy.copy(list(bn.nodes()))
    order.remove(target)
    
    #### EVIDENCE PROCESSING ####
    for E, e in evidence.items():
        for phi in _phi:
            if E in phi.scope:
                phi.reduce_factor(E,e)
        order.remove(E)

    #### ALGORITHM ####
    for var in order:
        phi_bold, traceback[var] = max_prod_eliminate_var(phi_bold, var)
    
    x_star = traceback_map(traceback)
    
    return bn.values(target)[np.argmax(marginal)]
    
        

def max_prod_eliminate_var(_phi, Z):
    relevant_factors = [f for f in temp_F if var in f.scope]
    irrelevant_factors = [f for f in temp_F if var not in f.scope]

    # mutliply all relevant factors
    psi = relevant_factors[0]
    for i in range(1,len(psi)):
        psi.multiply_factor(relevant_factors[i])
    
    _psi = deepcopy(psi)
    psi.maxout_var(var)
    irrelevant_factors.append(psi) # add sum-prod factor back in
    
    return irrelevant_factors, _psi
    

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
    var_dict = {} # maps variable string name to actual variable
    var_list = []
    for node in bn.nodes():
        for cpt_idx,cpt_val in enumerate(bn.cpt(node)):
            new_var = LpVariable(str(str(node)+'-'+str(cpt_idx)),0,1,LpInteger)
            var_list.append(new_var)
            var_dict[str(str(node)+'-'+str(cpt_idx))] = new_var
            node_var_dict[node].append(new_var)
            weight_list.append(-np.log(cpt_val))

    model += np.dot(weight_list,var_list) # minimizes -1*var*probability

    # constraint set 1
    # exactly one choice from each factor
    k = 0
    for rv in bn.nodes():
        cell = node_var_dict[rv]
        model += np.sum(cell) == 1, "Factor Sum Constraint" + str(k)
        k+=1

    factor_dict = dict([(rv, Factor(bn,rv)) for rv in bn.nodes()])
    # constraint set 2
    # intersection of factors must agree
    for rv1 in bn.nodes():
        for rv2 in bn.children(rv):
            f_rv1 = factor_dict[rv1]
            f_rv2 = factor_dict[rv2]
            # get their intersection
            #intersection_vars = set(bn.scope(rv1)) & set(bn.scope(rv2))
            #for var in intersection_vars:
            for value in bn.values(rv1):
                # get indices of var-value in f_rv1
                f_rv1_indices = f_rv1.value_indices(rv1,value)
                # get indices of var-value in f_rv2
                f_rv2_indices = f_rv2.value_indices(rv2,value)
                # get the associated model variables for f_rv1
                f_rv1_vars = [var_dict[str(str(rv1)+'-'+str(idx))] for idx in f_rv1_indices]
                f_rv2_vars = [var_dict[str(str(rv2)+'-'+str(idx))] for idx in f_rv2_indices]
                # create constraint that they must sum to the same thing
                model+= np.sum(f_rv1_vars)-np.sum(f_rv2_vars) == 0, 'Factor Agreement Constraint' + str(k)
            k+=1


    #add constraint set 3
    #all evidence variables set = 1
            
    model.solve()
    max_inference = dict([(v.name,v.varValue) for v in model.variables() if v.varValue == 1])
       return max_inference

    

    





