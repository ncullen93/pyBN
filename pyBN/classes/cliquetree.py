"""
******************
CliqueTree Class 
&
Clique Class
******************

This is a class for creating/manipulating Junction (Clique) Trees,
and performing inference over them. The advantage of clique trees
over traditional variable elimination over the original Bayesian 
network is that clique trees allow you to compute marginal
probabilities of MULTIPLE variables without having to run the
entire algorithm over. 

Therefore, if you have to query the Bayesian network many times 
-- and you want the exact marginal values -- then it might be 
best to use the clique tree data structure for inference.
If you need to compute the marginal distribution for all variables,
but you dont mind an approximate, a sampling algorithm is probably
the way to go.

In general, the junction tree algorithms generalize 
Variable Elimination to the efficient, simultaneous execution 
of a large class of queries.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



import numpy as np
import pandas as pd
import networkx as nx
import copy

from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor

class CliqueTree(object):
    """
    CliqueTree Class

    Let G be a chordal graph (ie. a graph such that no cycle includes more
    than three nodes), then a CliqueTree is a tree H such that each
    maximal clique C in G is a node in H.

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
        Ideally, the Factor class should be used as the
        cliques instead of the Clique class (because it's
        just a watered down version of the Factor class)        
        
        """
        self.bn = bn
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
        G = chordal_bn(self.bn)
        #G = self.bn.get_chordal_nx()
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
        factor_list = [Factor(bn,var) for var in self.bn.nodes()]
        for factor in factor_list:
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

class Clique(object):
    """
    Clique Class
    """

    def __init__(self, scope):
        """
        Instantiate a clique object

        Arguments
        ---------
        *scope* : a python set
            The set of variables in the cliques scope,
            i.e. the main var and its parents


        """
        self.scope=scope
        self.factors=[]
        self.psi = None # Psi should never change
        self.belief = None
        self.messages_received = []
        self.is_ready = False

    def send_initial_message(self, other_clique):
        """
        Send the first message to another clique.


        Arguments
        ---------
        *other_clique* : a different Clique object

        """
        psi_copy = copy.copy(self.psi)
        sepset = self.sepset(other_clique)
        sumout_vars = self.scope.difference(sepset)
        psi_copy.sumout_var_list(list(sumout_vars))
        psi_copy.cpt = psi_copy.cpt.loc[:,[c for c in psi_copy.cpt.columns if 'Prob' not in c]]
        psi_copy.cpt[str('Prob-Val-' + str(np.random.randint(0,1000000)))] = 1
        #print 'Init Msg: \n', psi_copy.cpt
        other_clique.messages_received.append(psi_copy)

        self.belief = copy.copy(self.psi)

    def sepset(self, other_clique):
        """
        The sepset of two cliques is the set of
        variables in the intersection of the two
        cliques' scopes.

        Arguments
        ---------
        *other_clique* : a Clique object

        """
        return self.scope.intersection(other_clique.scope)

    def compute_psi(self):
        """
        Compute a new psi (cpt) in order to 
        set the clique's belief.

        """
        if len(self.factors) == 0:
            print 'No factors assigned to this clique!'
            return None

        if len(self.factors) == 1:
            self.psi = copy.copy(self.factors[0])
        else:
            self.psi = max(self.factors, key=lambda x: len(x.cpt.columns))
            self.psi.merge_multiply(self.factors)
            self.belief = copy.copy(self.psi)

    def send_message(self, parent):
        """
        Send a message to the parent clique.

        Arguments
        ---------
        *parent* : a string
            The parent to which the message will
            be sent.

        """
        # First generate Belief = Original_Psi * all received messages
        if len(self.messages_received) > 0:
            # if there are messages received, mutliply them in to psi first
            if not self.belief:
                self.belief = copy.copy(self.psi)
            self.belief.merge_multiply(self.messages_received)
        else:
            # if there are no messages received, simply move on with psi
            if not self.belief:
                self.belief = copy.copy(self.psi)
        # generate message as belief with Ci - Sij vars summed out
        vars_to_sumout = list(self.scope.difference(self.sepset(parent)))
        message_to_send = copy.copy(self.belief)
        message_to_send.sumout_var_list(vars_to_sumout)
        parent.messages_received.append(message_to_send)

    def marginalize_over(self, target):
        """
        Marginalize the cpt (belief) over a target variable.

        Arguments
        ---------
        *target* : a string
            The target random variable.

        """
        self.belief.sumout_var_list(list(self.scope.difference({target})))

    def collect_beliefs(self):
        """        
        A root node (i.e. one that doesn't send a message) must run collect_beliefs
        since they are only collected at send_message()

        Also, we collect beliefs at the end of loopy belief propagation (approx. inference)
        since the main algorithm is just sending messages for a while.
        """
        if len(self.messages_received) > 0:
            self.belief = copy.copy(self.psi)
            for msg in self.messages_received:
                self.belief.cpt.merge(msg.cpt)
                #self.belief.normalize()
            self.belief.multiply_factor()
            print self.belief.cpt
            #self.belief.merge_multiply(self.messages_received)
            #self.belief.normalize()
            self.messages_received = []
        else:
            self.belief = copy.copy(self.belief)
            #print 'No Messages Received - Belief is just original Psi'