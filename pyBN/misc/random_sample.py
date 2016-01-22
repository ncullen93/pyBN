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

def random_sample(bn, n=100, evidence={}):
    """
    Take a random sample of "n" observations from a
    BayesNet object. This is essentially just a repeated
    forward sample algorithm that returns every sample.

    Parameters
    ----------
    *bn* : a BayesNet object from which to sample

    *n* : an integer
        The number of observations to take

    *evidence* : a dictionary, key=rv & value=instantiation
        Evidence to pass in

    Returns
    -------
    *sample_dict* : a list of samples, where each sample
        is a dictionary containing each variable as a key
        and its sampled value as the dictionary value

    Notes
    -----
    - Not tested

    """
    sample_dict = []
    weight_list = np.ones(n)

    parent_dict = dict([(var, bn.data[var]['parents']) for var in bn.V])

    for i in range(n):
        if i % (n/float(10)) == 0:
            if i > 0:
                print 'Sample: ' , i
        new_sample = {}
        for rv in rv_order:
            f = Factor(bn,rv)
            # reduce_factor by parent samples
            for p in parent_dict[rv]:
                f.reduce_factor(p,new_sample[p])
            # if rv in evidence, choose that value and weight
            if rv in evidence.keys():
                chosen_val = evidence[rv]
                weight_list[i] *= f.cpt[bn.data[rv]['vals'].index(evidence[rv])]
            # if rv not in evidence, sample as usual
            else:
                choice_vals = bn.data[rv]['vals']
                choice_probs = f.cpt
                chosen_val = np.random.choice(choice_vals, p=choice_probs)
                
            new_sample[rv] = chosen_val
        sample_dict.append(new_sample)

    return sample_dict


