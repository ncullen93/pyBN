"""
*******************
Factorization Class
*******************


"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
import pandas as pd
import networkx as nx
import copy
import json
from pybn.classes.bayesnet import BayesNet
from pybn.classes.factor import Factor


class Factorization(object):
    """
    
    DEPRECATED -- CODE NEEDS TO BE REMOVED &
    METHODS NEED TO BE INTEGRATED ASAP!!!

    """

    def __init__(self, BN, v_list=None):
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        """
        self.BN = BN
        if v_list:
            self.f_list = [Factor(BN,var) for var in v_list if var in self.BN.V]
            self.v_list = v_list
        else:
            self.f_list = [Factor(BN,var) for var in self.BN.V]
            self.v_list = BN.V
        self.temp_f_list = None # used so we can run multiple queries on same Factorization instantiation
        self.sol = None # most recent solution

    def shortest_path_map(self,
                          evidence={}):
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        """
        G = self.BN.get_networkx()
        

    def variable_elimination(self, 
                             target=[], 
                             evidence={}, 
                             order=None, 
                             marginal=True):
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        
        This is the main algorithmic driver of variable elimination --
        Both belief updating (marginal) and    belief revision (MAP) can
        be performed here with margin=True or marginal=False respectively.
        """

        self.temp_f_list = [Factor(self.BN,var) for var in self.BN.V]
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
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        """
        assignment={}
        for m in reversed(map_list):
            var = list(set(m.scope) - set(assignment.keys()))[0]
            m.reduce_factor_by_list([[k,v] for k,v in assignment.items() if k in m.scope and k!=var])
            assignment[var] = self.BN.data[var]['vals'][(np.argmax(m.cpt) / m.stride[var]) % m.card[var]]
        
        return assignment



