import numpy as np
import networkx as nx
import pandas as pd
from itertools import product
import copy

from pyBN.Inference import *
from pyBN.FileIO import *
from . import *

import time
import pdb

# %load_ext autoreload
# %autoreload 2

class BayesNet(object):

    def __init__(self):
        self.V = []
        self.E = []
        self.data = {}

        self.factorization = None
        self.sol = None

    def read(self, filepath):
        reader = BNio(self)
        reader.read(filepath)

    def write(self, filepath):
        writer = BNio(self)
        writer.write(filepath)

    def exact_inference(self, 
                        target=None, 
                        evidence=None, 
                        order=None, 
                        operation='marginal', 
                        algorithm='clique_tree'):
        """
        This is the wrapper function for user interaction w/ exact inference.

        Methods:
            1. *marginal* - computes marginal and conditional queries w/ or w/o evidence
                Algorithms:
                    1. *sum_product* - traditional sum-product variable elimination of factors
                    2. *clique_tree* - Shafer-Shenoy message passing algorithm over a clique tree
                    3. *hugin* - Hugin message passing over a clique tree - NOT IMPLEMENTED YET

            2. *map* - computes MAP inference (i.e. most probable world state w/ or w/o evidence)
                Algorithms: NOT IMPLEMENTED YET

        """
        inference = ExactInference(self, operation)
        
        if algorithm == 'sum_product':
            inference.sum_product_ve(target, evidence, order)
        else:
            inference.clique_tree_bp(target, evidence)

    def approx_inference(self, 
                         target=None,
                         evidence=None,
                         order=None,
                         operation='marginal',
                         algorithm='loopy'):
        """
        This is the wrapper function for user interaction w/ approximate inference.

        Methods:
            1. *marginal* - computes marginal and conditional queries w/ or w/o evidence
                Algorithms:
                    1. *loopy* - loopy belief propagation over cluster graph
        """
        inference = ApproxInference(self)

        if algorithm == 'loopy':
            inference.loopy_belief_propagation(target, evidence)

    #####################################################################
    #####################################################################
    ###################### UTILITY METHODS ##############################
    #####################################################################
    #####################################################################

    def get_cpt(self, var):
        """
        This function returns the CPT (only) of a variable as a DataFrame.

        This is a helpful function in the intitialization of Factor objects
        because factor cpt's are stored as Pandas DataFrames 
        (instead of flattene arrays as recommended in Koller)
        """
        data = copy.copy(self.data)
        if len(data[var]['parents']) > 0:
            val_iter = [data[p]['vals'] for p in data[var]['parents']]
            val_iter.append(data[var]['vals'])
            val_comb = map(list,list(product(*val_iter))) # from itertools
            
            prob_ravel = np.ravel(np.array(data[var]['cprob']))
            
            c = zip(*val_comb)
            c.append(tuple(prob_ravel))
            cpt = pd.DataFrame(c).T
            
            columns = copy.copy(data[var]['parents'])
            columns.append(var)
            columns.append(str('Prob-'+var))
            cpt.columns = columns
        else:
            cpt = pd.DataFrame(zip(data[var]['vals'],data[var]['cprob']))
            cpt.columns = [var, str('Prob-'+ var)]
        return cpt

    def get_networkx(self):
        """
        This function returns ONLY the network structure of the BN
        in networkx form - i.e. there is no data/probabilities associated.
        """
        G = nx.DiGraph()
        edge_list = self.E
        G.add_edges_from(edge_list)
        return G

    def get_sp_networkx(self):
        """
        This function returns a weighted Digraph based on a 
        topological sort of the original BN. Solving the shortest
        path on this graph is equivalent to Belief Revision
        """
        OG = self.get_networkx()

        sp_sort = dict([(n,0) for n in OG.nodes() if len(OG.predecessors(n))==0])


        node_list = sp_sort.keys()

        while node_list:
            node = node_list.pop(0)
            for neighbor in OG.successors(node):
                if neighbor in sp_sort:
                    if sp_sort[node] >= sp_sort[neighbor]:
                        sp_sort[neighbor] = sp_sort[node]+1
                else:
                    sp_sort[neighbor] = sp_sort[node]+1
                node_list.append(neighbor)
        
        G=nx.DiGraph()
        G.add_node('source')
        G.add_node('sink')
        for i in range(max(sp_sort.values())+1):
            new_nodes = []
            nodes = [n for n in sp_sort if sp_sort[n]==i]
            node_vals = [self.data[n]['vals'] for n in nodes]
            print nodes
            node_vals.append([str(i)])

            val_combs = map(list,list(product(*node_vals)))
            for val in val_combs:
                new_node = "-".join(val)
                G.add_node(new_node)

                if i == 0:
                    G.add_edge('source',new_node)
                else:
                    previous_nodes = [n for n in G.nodes() if str(i-1) in n]
                    for previous_node in previous_nodes:
                        G.add_edge(previous_node,new_node)
                if i == max(sp_sort.values()):
                    G.add_edge(new_node,'sink')
        return G







    def get_moralized_edge_list(self):
        """
        This function creates the moral of a BN - i.e. it
        adds an edge between each of the parents of each node if
        there isn't already an edge between them.

        Returns:
            1. *e_list* - a list of lists (edges)
        """
        e_list = copy.copy(self.E)
        for node in self.V:
            parents = self.data[node]['parents']
            for p1 in parents:
                for p2 in parents:
                    if p1!=p2 and [p1,p2] not in e_list and [p2,p1] not in e_list:
                        e_list.append([p1,p2])
        return e_list


    def get_chordal_nx(self,v=None,e=None):
        """
        This function creates a chordal graph - i.e. one in which there
        are no cycles with more than three nodes.

        Can supply a v list and e list for chordal graph of any random graph..

        We start from the moral graph, so if that is already chordal then it
        will return that.

        Note: 
            -Algorithm from Cano & Moral 1990 ->
            'Heuristic Algorithms for the Triangulation of Graphs'
        """
        chordal_E = self.get_moralized_edge_list() # start with moral graph

        # if moral graph is already chordal, no need to alter it
        if not self.is_chordal(chordal_E):            
            temp_E = copy.copy(chordal_E)
            temp_V = []

            # if v and e is supplied, skip all the rest
            if v and e:
                chordal_E = copy.copy(e)
                temp_E = copy.copy(chordal_E)
                temp_V = copy.copy(v)
            else:
                temp_G = nx.Graph()
                temp_G.add_edges_from(chordal_E)
                degree_dict = temp_G.degree()
                temp_V = sorted(degree_dict, key=degree_dict.get)
            #print temp_V
            for v in temp_V:
                #Add links between the pairs nodes adjacent to Node i
                #Add those links to chordal_E and temp_E
                adj_v = set([n for e in temp_E for n in e if v in e and n!=v])
                for a1 in adj_v:
                    for a2 in adj_v:
                        if a1!=a2:
                            if [a1,a2] not in chordal_E and [a2,a1] not in chordal_E:
                                chordal_E.append([a1,a2])
                                temp_E.append([a1,a2])
                # remove Node i from temp_V and all its links from temp_E 
                temp_E2 = []
                for edge in temp_E:
                    if v not in edge:
                        temp_E2.append(edge)
                temp_E = temp_E2

        g = nx.Graph()
        g.add_edges_from(chordal_E)
        return g

    def is_chordal(self, edge_list=None):
        """
        Returns true if the graph is chordal/triangulated
        """
        if not edge_list:
            edge_list = self.E
        G = nx.Graph()
        G.add_edges_from(edge_list)
        return nx.is_chordal(G)