"""
****************
CliqueTree Class
****************

This is a class for creating/manipulating Clique Trees, and
performing inference over them. The advantage of clique trees
over traditional variable elimination over the original Bayesian 
network is that clique trees allow you to compute marginal
probabilities of MULTIPLE variables without having to run the
entire algorithm over. Therefore, if you have to query the Bayesian
network many times, it might be best to use the clique tree data
structure for inference.

See Clique class

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



import numpy as np
import pandas as pd
import networkx as nx
import copy

from pybn.classes.bayesnet import BayesNet
from pybn.classes.factor import Factor

class CliqueTree(object):
    """
    CliqueTree Class

    """

    def __init__(self, bn):
        """
        Instantiate a CliqueTree object.

        Arguments
        ---------
        *bn*: a BayesNet object

        Returns
        -------


        Notes
        -----

        
        Let G be a chordal graph (ie. a graph such that no cycle includes more
        than three nodes), then a CliqueTree is a tree H such that each
        maximal clique C in G is a node in H.
        """
        self.bn = bn
        self.V = {} # key = cluster index, value = Clique objects
        self.E = []
        self.G=None # networkx graph
        self.initialize_tree()

    def initialize_tree(self):
        """
        Initialize the structure of a clique tree, using
        the following steps:
            - Moralize graph (i.e. marry parents)
            - Triangulate graph (i.e. make graph chordal)
            - Get max cliques (i.e. community/clique detection)
            - Max spanning tree over sepset cardinality (i.e. create tree)
        
        """
        self.V = {}
        self.E  =[]
        # get chordal/triangulated graph
        G = self.bn.get_chordal_nx()
        # get max cliques from chordal
        max_cliques = reversed(list(nx.chordal_graph_cliques(G)))
        for clique in max_cliques:
            self.V[len(self.V)] = Clique(set(clique))
        # find edges used maximum spanning tree
        new_e_list = []
        for i in range(len(self.V)):
            for j in range(len(self.V)):
                if i!=j:
                    intersect_cardinality = len(self.V[i].sepset(self.V[j]))
                    new_edge = (i,j,-1*intersect_cardinality)
                    new_e_list.append(new_edge)

        new_G = nx.Graph()
        new_G.add_weighted_edges_from(new_e_list)
        mst_G = nx.minimum_spanning_tree(new_G)
        self.E = mst_G.edges(data=False)
        self.G = mst_G

        self.assign_factors()
        for clique in self.V.values():
            clique.compute_psi()

    def assign_factors(self):
        """
        This clearly needs to be changed       

        """
        factorization = Factorization(self.bn)
        for f in factorization.f_list:
            assigned=False
            for v in self.V.values():
                if f.scope.issubset(v.scope):
                    if assigned==False:
                        v.factors.append(f)
                        assigned=True

    def message_passing(self, target=None, evidence=None, downward_pass=True):
        """
        Perform Message Passing (Belief Propagation) over a clique tree. This
        includes an Upward Pass as shown in Koller p.353 along with
        Downward Pass (Calibration) from Koller p.357 if target is list.

        The result is a marginal distribution over the target rv(s).


        Arguments
        ---------
        *target* : a string or a list of strings
            The variables for which the marginal probabilities
            are to be computed.

        *evidence* : a dictionary, where
            key = rv and value = rv's instantiation

        Returns
        -------
        None

        Effects
        -------
        - Sends messages

        Notes
        -----

        """
        # 1: Moralize the graph
        # 2: Triangluate
        # 3: Build a clique tree using max spanning
        # 4: Propagation of probabilities using message passing

        # creates clique tree and assigns factors, thus satisfying steps 1-3
        ctree = copy.copy(self)
        G = ctree.G
        #cliques = copy.copy(ctree.V)

        # select a clique as root where target is in scope of root
        root=np.random.randint(0,len(ctree.V))
        if target:
            root = [node for node in G.nodes() if target in ctree.V[node].scope][0]

        tree_graph = nx.dfs_tree(G,root)
        clique_ordering = list(nx.dfs_postorder_nodes(tree_graph,root))

        # SEND MESSAGES UP THE TREE FROM THE LEAVES TO THE SINGLE ROOT
        for i in clique_ordering:
            clique = ctree.V[i]
            for j in tree_graph.predecessors(i):
                clique.send_message(ctree.V[j])
            # if root node, collect its beliefs
            if len(tree_graph.predecessors(i)) == 0:
                ctree.V[root].collect_beliefs()

        if downward_pass:
            # if target is a list, run downward pass
            new_ordering = list(reversed(clique_ordering))
            for j in new_ordering:
                clique = ctree.V[j]
                for i in tree_graph.successors(j):
                    clique.send_message(ctree.V[i])
                # if leaf node, collect its beliefs
                if len(tree_graph.successors(j)) == 0:                    
                    ctree.V[j].collect_beliefs()

        self.bn.ctree = self

        # beliefs hold the answers