import numpy as np
import networkx as nx
import pandas as pd
import time
from pyBN.Models.FastFactor import FastFactor

class Sampler:
    """
    DEPRECATED -- NEEDS TO BE REMOVED &
    METHODS NEED TO BE INTEGRATED ASAP!!!
    


    """

    def __init__(self, BN):
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
        self.sample_dict = []
        self.forward_counter = {}
        self.gibbs_counter = {}
        self.lw_counter = {}

    def random_sample(self, n=1, evidence={}):
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
        sample_dict = []
        weight_list = np.ones(n)

        G = self.BN.get_networkx()
        rv_order = nx.topological_sort(G)

        #factor_dict = dict([(var,FastFactor(self.BN, var)) for var in self.BN.V])
        parent_dict = dict([(var, G.predecessors(var)) for var in self.BN.V])

        for i in range(n):
            if i % (n/float(10)) == 0:
                if i > 0:
                    print 'Sample: ' , i
            new_sample = {}
            for rv in rv_order:
                f = FastFactor(self.BN,rv)
                # reduce_factor by parent samples
                for p in parent_dict[rv]:
                    f.reduce_factor(p,new_sample[p])
                # if rv in evidence, choose that value and weight
                if rv in evidence.keys():
                    chosen_val = evidence[rv]
                    weight_list[i] *= f.cpt[self.BN.data[rv]['vals'].index(evidence[rv])]
                # if rv not in evidence, sample as usual
                else:
                    choice_vals = self.BN.data[rv]['vals']
                    choice_probs = f.cpt
                    chosen_val = np.random.choice(choice_vals, p=choice_probs)
                    
                new_sample[rv] = chosen_val
            sample_dict.append(new_sample)

        self.sample_dict = sample_dict

    def forward_sample(self, n=1000):
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        
        
        Take *n* forward samples of the network and stores the counts in
        self.sample_dict for easy manipulation.

        *Hoeffding Bound*:
            For an additive bound *epsilon* on error (i.e. T_D \in (p-eps, p+eps))
            with probability 1-*delta*, we need n >= ln(2/delta) / 2*epsilon^2
        *Chernoff Bound*:
            For a multiplicative bound *epsilon* on error (i.e. T_d \in (p(1-eps), p(1+eps))) 
            with probability 1-*delta*, we need n >= 3 * ln(2/delta) / p*eps^2
        """
        sample_dict = {}

        G = self.BN.get_networkx()
        rv_order = nx.topological_sort(G)

        #factor_dict = dict([(var,FastFactor(self.BN, var)) for var in self.BN.V])
        parent_dict = dict([(var, G.predecessors(var)) for var in self.BN.V])

        for var in self.BN.V:
            sample_dict[var] = {}
            for val in self.BN.data[var]['vals']:
                sample_dict[var][val] = 0

        for i in range(n):
            if i % (n/float(10)) == 0:
                print 'Sample: ' , i
            new_sample = {}
            for rv in rv_order:
                f = FastFactor(self.BN,rv)
                for p in parent_dict[rv]:
                    f.reduce_factor(p,new_sample[p])
                choice_vals = self.BN.data[rv]['vals']
                choice_probs = f.cpt
                chosen_val = np.random.choice(choice_vals, p=choice_probs)

                sample_dict[rv][chosen_val] += 1
                new_sample[rv] = chosen_val

        for rv in sample_dict:
            for val in sample_dict[rv]:
                sample_dict[rv][val] = int(sample_dict[rv][val]) / float(n)
        self.forward_counter=sample_dict

    def gibbs_sample(self, n=5000, burn=500):
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        
        
        Use Markov Chain Monte Carlo method for Gibbs sampling. 

        Starts by drawing a initial random sample of each rv in the network,
        then samples each rv one-by-one while conditioning on the values of
        all other rv's. Re-sampling each rv once counts as ONE sample.
        """
        G=self.BN.get_networkx()
        bn=self.BN
        counter={}
        for rv in bn.V:
            counter[rv]={}
            for val in bn.data[rv]['vals']:
                counter[rv][val] = 0

        state = {}
        for rv in bn.V:
            state[rv] = np.random.choice(bn.data[rv]['vals']) # uniform sample

        for i in range(n):
            if i % (n/float(10)) == 0:
                print 'Sample: ' , i
            for rv in bn.V:
                # get possible values conditioned on everything else
                parents = G.predecessors(rv)
                # no parents - prior
                if len(parents) == 0:
                    choice_vals = bn.data[rv]['vals']
                    choice_probs = bn.data[rv]['cprob']
                # has parent - filter cpt
                else:
                    f = FastFactor(bn,rv)
                    for p in parents:
                        f.reduce_factor(p,state[p])
                    choice_vals = bn.data[rv]['vals']
                    choice_probs = f.cpt
                # sample over remaining possibilities
                chosen_val = np.random.choice(choice_vals,p=choice_probs)
                state[rv]=chosen_val
            # update counter dictionary
            if i > burn:
                for rv,val in state.items():
                    counter[rv][val] +=1

        for rv in counter:
            for val in counter[rv]:
                counter[rv][val] = round(int(counter[rv][val]) / float(n-burn),4)
        self.gibbs_counter=counter

    def lw_sample(self, n=1000, evidence={}):
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        
        
        Perform likelihood weighted particle sampling over network
        with evidence.

        Arguments:
            1. *evidence* - a dictionary where key=rv, val=instantiation


        """
        sample_dict = {}
        weight_list = np.ones(n)

        G = self.BN.get_networkx()
        rv_order = nx.topological_sort(G)

        #factor_dict = dict([(var,FastFactor(self.BN, var)) for var in self.BN.V])
        parent_dict = dict([(var, G.predecessors(var)) for var in self.BN.V])

        for var in self.BN.V:
            sample_dict[var] = {}
            for val in self.BN.data[var]['vals']:
                sample_dict[var][val] = 0

        for i in range(n):
            if i % (n/float(10)) == 0:
                print 'Sample: ' , i
            new_sample = {}
            for rv in rv_order:
                f = FastFactor(self.BN,rv)
                # reduce_factor by parent samples
                for p in parent_dict[rv]:
                    f.reduce_factor(p,new_sample[p])
                # if rv in evidence, choose that value and weight
                if rv in evidence.keys():
                    chosen_val = evidence[rv]
                    weight_list[i] *= f.cpt[self.BN.data[rv]['vals'].index(evidence[rv])]
                # if rv not in evidence, sample as usual
                else:
                    choice_vals = self.BN.data[rv]['vals']
                    choice_probs = f.cpt
                    chosen_val = np.random.choice(choice_vals, p=choice_probs)
                    
                new_sample[rv] = chosen_val
            # weight the choice by the evidence likelihood    
            for rv in new_sample:
                sample_dict[rv][new_sample[rv]] += 1*weight_list[i]

        weight_sum = sum(weight_list)
        
        for rv in sample_dict:
            for val in sample_dict[rv]:
                sample_dict[rv][val] /= weight_sum
                sample_dict[rv][val] = round(sample_dict[rv][val],4)
        self.lw_counter=sample_dict







