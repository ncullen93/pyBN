"""
**************
BayesNet Class
**************

Overarching class for Discrete Bayesian Networks.


Design Specs
------------

- Bayesian Network -

    - F -
        key:
            - rv -
        values:
            - children -
            - parents -
            - values -
            - cpt -

    - V -
        - list of rvs -

    - E -
        key:
            - rv -
        values:
            - list of rv's children -
Notes
-----
- Edges can be inferred from Factorization, but Vertex values 
    must be specified.
"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



import numpy as np
from itertools import product
import copy
from pyBN.utils.class_equivalence import are_class_equivalent
from pyBN.utils.topsort import topsort

class BayesNet(object):
    """
    Overarching class for Bayesian Networks

    """

    def __init__(self,E=None,value_dict=None):
        """
        Initialize the BayesNet class.

        Arguments
        ---------
        *V* : a list of strings - vertices in topsort order
        *E* : a dict, where key = vertex, val = list of its children
        *F* : a dict, 
            where key = rv, 
            val = another dict with
                keys = 
                    'parents', 
                    'values', 
                    'cpt'

        *V* : a dict        

        Notes
        -----
        
        """
        if E is not None:
            assert (value_dict is not None), 'Must set values if E is set.'
            self.set_structure(E,value_dict)
        else:
            self.V = list
            self.E = dict
            self.F = dict

    def __eq__(self, y):
        """
        Tests whether two Bayesian Networks are
        equivalent - i.e. they contain the same
        node/edge structure, and equality of
        conditional probabilities.
        """
        return are_class_equivalent(self, y)

    def __hash__(self):
        """
        Allows BayesNet objects to be used
        as keys in a dictionary (i.e. hashable)
        """
        return hash((str(self.V),str(self.E)))

    def add_node(self, rv, values=None):
        self.V.append(rv)
        if values is not None:
            self.F[rv] = {'cpt':[],'parents':[],'values':values}
        else:
            self.F[rv] = {'cpt':[],'parents':[],'values':[]}

    def add_edge(self, u, v):
        if not self.has_node(u):
            self.add_node(u)
        if not self.has_node(v):
            self.add_node(v)
        self.E[u].append(v)
        self.V = topsort(self.E)

    def set_data(self, rv, data):
        assert (isinstance(data, dict)), 'data must be dictionary'
        self.F[rv] = data

    def set_cpt(self, rv, cpt):
        self.F[rv]['cpt'] = cpt

    def set_parents(self, rv, parents):
        self.F[rv]['parents'] = parents

    def set_values(self, rv, values):
        self.F[rv]['values'] = values

    def nodes(self):
        for v in self.V:
            yield v

    def node_idx(self, rv):
        try:
            return self.V.index(rv)
        except ValueError:
            return -1

    def has_node(self, rv):
        return rv in self.V

    def has_edge(self, u, v):
        return u in self.E[v]

    def edges(self):
        for u in self.nodes():
            for v in self.children(u):
                yield (u,v)

    def moralized_edges(self):
        """
        Moralized graph is the original graph PLUS
        an edge between every set of common effect
        structures -
            i.e. all parents of a node are connected.

        This function has be validated.

        Returns
        -------
        *e* : a python list of parent-child tuples.

        """
        e = set()
        for u in self.nodes():
            for p1 in self.parents(u):
                e.add((p1,u))
                for p2 in self.parents(u):
                    if p1!=p2 and (p2,p1) not in e:
                        e.add((p1,p2))
        return list(e)


    def scope_size(self, rv):
        return len(self.F[rv]['parents'])+1

    def num_nodes(self):
        return len(self.V)

    def cpt(self, rv):
        return self.F[rv]['cpt']

    def card(self, rv):
        return len(self.F[rv]['values'])

    def scope(self, rv):
        scope = [rv]
        scope.extend(self.F[rv]['parents'])
        return scope

    def parents(self, rv):
        return self.F[rv]['parents']

    def children(self, rv):
        return self.E[rv]

    def degree(self, rv):
        return len(self.parents(rv)) + len(self.children(rv))

    def values(self, rv):
        return self.F[rv]['values']

    def value_idx(self, rv, val):
        try:   
            return self.F[rv]['values'].index(val)
        except ValueError:
            print "Value Index Error"
            return -1

    def stride(self, rv, n):
        if n==rv:
            return 1
        else:
            card_list = [self.card(rv)]
            card_list.extend([self.card(p) for p in self.parents(rv)])
            n_idx = self.parents(rv).index(n)
            return np.prod(card_list[0:n_idx])

    def flat_cpt(self):
        """
        Return all cpt values in the BN as a flattened
        numpy array ordered by bn.nodes() - i.e. topsort
        """
        cpt = np.array([val for rv in self.nodes() for val in self.cpt(rv)])
        return cpt

    def cpt_value_indices(self, target, val_dict):
        """
        Get the index of the CPT which corresponds
        to a dictionary of rv=val sets. This can be
        used for parameter learning to increment the
        appropriate cpt frequency value based on
        observations in the data.

        """
        stride = dict([(n,self.stride(target,n)) for n in val_dict])
        card = dict([(n,self.card(n)) for n in val_dict])
        idx_set = range(len(self.cpt(target)))

        for rv, val in val_dict.items():
            rv_stride = stride[rv]
            rv_card = card[rv]
            value_idx = self.bn.value_idx(rv,value)

            value_indices = []
            idx=rv_stride*value_idx         
            while idx < len(self.cpt):
                value_indices.extend(range(idx,idx+rv_stride))
                idx+=rv_card*rv_stride
            idx_set.intersection(set(value_indices))

        return idx_set

    def set_structure(self, edge_dict, value_dict):
        """
        Set the structure of a BayesNet object. This
        function is mostly used to instantiate a BN
        skeleton after structure learning algorithms.

        See "structure_learn" folder & algorithms

        Arguments
        ---------
        *edge_dict* : a dictionary,
            where key = rv,
            value = list of rv's children
            NOTE: THIS MUST BE DIRECTED ALREADY!

        *value_dict* : a dictionary,
            where key = rv,
            value = list of rv's possible values

        Returns
        -------
        None

        Effects
        -------
        - sets self.V in topsort order from edge_dict
        - sets self.E
        - creates self.F structure and sets the parents

        Notes
        -----

        """
        from pyBN.utils.topsort import topsort

        self.V = topsort(edge_dict)
        self.E = edge_dict
        self.F = dict([(rv,{}) for rv in self.nodes()])
        for rv in self.nodes():
            self.F[rv] = {
                'parents':[p for p in self.nodes() if rv in self.children(p)],
                'cpt': [],
                'values': value_dict[rv]
            }

    def adj_list(self):
        """
        Returns adjacency list of lists, where
        each list element is a vertex, and each sub-list is
        a list of that vertex's neighbors.
        """
        adj_list = [[] for _ in self.V]
        vi_map = dict((self.V[i],i) for i in range(len(self.V)))
        for u,v in self.edges():
            adj_list[vi_map[u]].append(vi_map[v])
        return adj_list


    


    
