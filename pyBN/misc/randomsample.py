"""
******************
Random Sample Code
******************

Generate a random sample dataset from a known Bayesian Network,
with or without evidence.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.classes.factor import Factor
import networkx as nx

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
            f = Factor(self.BN,rv)
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


