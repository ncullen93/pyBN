import numpy as np
import pandas as pd
import networkx as nx
import copy
import json
import BayesNet
from FastFactor import FastFactor


class Factorization(object):
    """
    Class for Factorization.

    A factorization is a product of a collection of factors.

    Attributes
    ----------


    Methods
    -------


    Notes
    -----

    """

    def __init__(self, BN, v_list=None):
        self.BN = BN
        if v_list:
            self.f_list = [FastFactor(BN,var) for var in v_list if var in self.BN.V]
            self.v_list = v_list
        else:
            self.f_list = [FastFactor(BN,var) for var in self.BN.V]
            self.v_list = BN.V
        self.temp_f_list = None # used so we can run multiple queries on same Factorization instantiation
        self.sol = None # most recent solution

    def shortest_path_map(self,
                          evidence={}):
        G = self.BN.get_networkx()
        

    def variable_elimination(self, 
                             target=[], 
                             evidence={}, 
                             order=None, 
                             marginal=True):
        """
        This is the main algorithmic driver of variable elimination --
        Both belief updating (marginal) and    belief revision (MAP) can
        be performed here with margin=True or marginal=False respectively.
        """

        self.temp_f_list = [FastFactor(self.BN,var) for var in self.BN.V]
        map_list = []

        #### ORDER HANDLING ####
        if not order:
            order = copy.copy(self.BN.V)
            if isinstance(target,list):
                for t in target:
                    order.remove(t)
            else: 
                order.remove(target)
        if isinstance(target,list):
            for t in target:
                if target in order:
                    order.remove(t)
        else:
            if target in order:
                order.remove(target)

        ##### EVIDENCE #####

        if len(evidence)>0:
            assert isinstance(evidence, dict), 'Evidence must be Dictionary'
            temp=[]
            for obs in evidence.items():
                for f in self.temp_f_list:
                    if len(f.scope)>1 or obs[0] not in f.scope:
                        temp.append(f)
                    if obs[0] in f.scope:
                        f.reduce_factor(obs[0],obs[1])
                order.remove(obs[0])
            self.temp_f_list=temp

        #### ALGORITHM ####
        for var in order:
            relevant_factors = [f for f in self.temp_f_list if var in f.scope]
            irrelevant_factors = [f for f in self.temp_f_list if var not in f.scope]
            
            # mutliply all relevant factors
            fmerge = relevant_factors[0]
            for i in range(1,len(relevant_factors)):
                fmerge.multiply_factor(relevant_factors[i])
            ## only difference between marginal and map
            if marginal==False:
                map_list.append(copy.deepcopy(fmerge))
                fmerge.maxout_var(var)
            else:
                fmerge.sumout_var(var) # remove var from factor
                

            irrelevant_factors.append(fmerge) # add sum-prod factor back in
            self.temp_f_list = irrelevant_factors
        
        if marginal==False:
            self.sol=self.traceback_MAP(map_list)
            self.sol.update(evidence)
            print json.dumps(self.sol,indent=2)
            self.val=round(self.temp_f_list[0].cpt[0],4)
            #self.traceback_MAP()
        else:
            marginal = self.temp_f_list[0]
            # multiply final factors in factor_list
            if len(self.temp_f_list) > 1:
                for i in range(1,len(self.temp_f_list)):
                    marginal.multiply_factor(self.temp_f_list[i])
            marginal.normalize()

            self.sol=marginal.cpt

    def traceback_MAP(self,map_list):
        assignment={}
        for m in reversed(map_list):
            var = list(set(m.scope) - set(assignment.keys()))[0]
            m.reduce_factor_by_list([[k,v] for k,v in assignment.items() if k in m.scope and k!=var])
            assignment[var] = self.BN.data[var]['vals'][(np.argmax(m.cpt) / m.stride[var]) % m.card[var]]
        
        return assignment



class Factor:
    """
    DEPRECATED -- REMOVE FROM CODE BASE ASAP & REPLACE WITH "FastFactor"

    """
    def __init__(self, BN, var, init_to_one=False):
        self.BN = BN
        if init_to_one:
            self.var = var
            self.scope = {var}
            init_col = 'Init-Prob-' + str(np.random.randint(0,100000))
            self.cpt = pd.DataFrame({init_col:1, var:self.BN.data[var]['vals']})
        else:
            self.var = var
            self.scope = {var}.union(set(BN.data[var]['parents']))
            self.cpt = BN.get_cpt(var)

    def merge_multiply(self, factor_list):
        """
        Calls merge_factor() and multiply_factor() together
        """
        for factor in factor_list:    
            self.cpt.merge(factor.cpt)
            self.scope = self.scope.union(factor.scope)
        #print factor.cpt
        self.multiply_factor()

    def merge_factor(self, other_factor):
        self.cpt = self.cpt.merge(other_factor.cpt)
        
        #self.multiply_factor()

    def multiply_factor(self):
        multiply_cols = [c for c in self.cpt.columns if 'Prob' in c or 'Multiply' in c]
        new_col = 'Multiply ' + str(np.random.randint(0,100000))
        self.cpt[new_col] = self.cpt.apply(lambda row: np.prod([row[c] for c in multiply_cols]), axis=1)
        keep_col = self.cpt[new_col]
        self.cpt = self.cpt.loc[:,[c for c in self.cpt.columns if 'Prob-' not in c and 'Multiply' not in c]]
        self.cpt[new_col] = keep_col
        #print self.cpt

    def update_evidence(self, obs):
        self.cpt = self.cpt[self.cpt.loc[:,obs[0]] == obs[1]]
        self.cpt = self.cpt.drop(obs[0],axis=1)
        if obs[0] in self.scope:
            self.scope.remove(obs[0])

    def sumout_var(self, var):
        #print self.scope
        self.scope = self.scope.difference({var})
        #print self.scope
        grouped_df = self.cpt.groupby(list(self.scope))
        self.cpt = grouped_df.sum().reset_index()
        #print self.cpt

    def sumout_var_list(self, var_list):
        for var in var_list:
            self.scope = self.scope.difference({var})
            #print self.scope
            grouped_df = self.cpt.groupby(list(self.scope))
            self.cpt = grouped_df.sum().reset_index()

    def normalize(self):
        col = [c for c in self.cpt.columns if c not in self.BN.V][0]
        self.cpt[col] = self.cpt[col] / float(self.cpt[col].sum())