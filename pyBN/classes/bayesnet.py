"""
**************
BayesNet Class
**************


"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



import numpy as np
import networkx as nx
import pandas as pd
from itertools import product
import copy

import pyBN.readwrite

import time
import pdb

# %load_ext autoreload
# %autoreload 2

class BayesNet(object):
    """
    Overarching class for Bayesian Networks


    Attributes
    ----------
    *V* : a list
        The container for vertices (i.e. random variables)

    *E* : a list of tuples ?
        The container for edges (i.e. conditional dependencies)

    *data* : a dictionary
        The container for probability values

        First-Level Keys:

            vertex name : another dictionary

            Second-Level Keys:

                "numoutcomes" : an integer
                    The number of outcomes an RV has.

                "vals" : a list
                    The list of instantiations (values) an RV has.

                "parents" : a list or None
                    The list of the parents' names

                "children": a list or None
                    The list of the childrens' names

                "cprob" : a nested python list
                    The probability values for every combination
                    of parent(s)-self values

    User Methods
    ------------

    *read* : read a BayesNet object from a file

    *write* : write a BayesNet object to a file

    *set_structure* : set the BayesNet structure from an edge dictionary.

    
    Utility Methods
    ---------------

    *get_networkx* : get networkx representation of BayesNet object

    *get_sp_networkx* : get weighted/expanded networkx representation of
        BayesNet object

    *get_moralized_edge_list* : get edge list of moralized graph

    *get_chordal_nx* : get chordal networkx representation

    *is_chordal* : test whether a graph is chordal


    Notes
    -----


    """

    def __init__(self):
        """
        Initialize the BayesNet class

        Parameters
        ----------
        None

        Returns
        -------
        None

        Effects
        -------
        - initializing the attributes

        Notes
        -----
        - perhaps should add the capability of initializing with data (i.e. edge list, etc)
        
        """
        self.V = []
        self.E = []
        self.data = {}

        #self.factorization = None
        #self.sol = None

    ###################### USER METHODS ##############################

    def read(self, path):
        """
        Read a file and create a BayesNet object from it

        Currently supported extensions:
            - .bif
            - .bn (json format - see above)

        Parameters
        ----------
        *path* : a string
            The relative or absolute file path to the file, INCLUDING the
            extension. 


        Returns
        -------
        None

        Effects
        -------
        - populates self.V, self.E, self.data with info from the file

        Notes
        -----
        - any other common extensions to read from besides .bif?
        
        """
        pybn.readwrite.read(path)

    def write(self, path):
        """
        Write a BayesNet object to file.

        Currently supported extensions:
            - .bn (json format)


        Parameters
        ----------
        *path* : a string

        Returns
        -------
        None

        Effects
        -------
        - creates a new file in user's system

        Notes
        -----

        
        """
        pybn.readwrite.write(path)

    def set_structure(self, edge_dict, card_dict):
        """
        Set the node/edge structure of this BayesNet object
        from the given edge_dict. If the current BayesNet's structure
        is empty, it will be initialized. If it is already initialized,
        it will be overwritten. If there is any parameter information, 
        it will be cleared.

        Arguments
        ---------
        *edge_dict* : a dictionary where key = node, value = python list of its neighbors

        *card_dict* : a dictionary where key = node, value = python list of its values

        Returns
        -------
        None

        Effects
        -------
        - Sets the structure information

        Notes
        -----
        Values to set:
            "numoutcomes" : an integer
                        The number of outcomes an RV has.

            "vals" : a list
                The list of instantiations (values) an RV has.

            "parents" : a list
                The list of the parents' names

            "children": a list
                The list of the childrens' names

            "cprob" : a nested python list
                The probability values for every combination
                of parent(s)-self values
        """
        data = dict((rv,\
            {'numoutcomes':card_dict[rv],
            'vals':range(card_dict[rv]),
            'parents':[i for i in edge_dict.keys() if rv in edge_dict[i]],
            'children':edge_dict[rv],'cprob':[]}) for rv in edge_dict.keys())        
        
        self.data = data
        self.E = [(i,j) for i in edge_dict.keys() for j in edge_dict[i]]


    ###################### UTILITY METHODS ##############################

    def get_networkx(self):
        """
        This function returns ONLY the network structure of the BN
        in networkx form - there is no data/probabilities associated.

        This is definitely used for drawing, and should also be the case. But
        it is also used to find the topological sort of the BN, which is completely
        unnecessary.

        Parameters
        ----------
        None

        Returns
        -------
        *G* : a networkx DiGraph object

        Notes
        -----


        """
        G = nx.DiGraph()
        edge_list = self.E
        G.add_edges_from(edge_list)
        return G

    def get_sp_networkx(self):
        """
        This function returns ONLY the network structure of the BN
        in networkx form -  there is no data/probabilities associated.

        Parameters
        ----------
        None

        Returns
        -------
        *G* : a networkx digraph object

        Effects
        -------
        None

        Notes
        -----
        This isn't really used anywhere.

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

        Parameters
        ----------
        None

        Returns
        -------
        *e_list* : a list of lists contains the edges of moralized graph

        Notes
        -----
        Where is this used?

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

        Can supply a v_list and e_list for chordal graph of any random graph..

        We start from the moral graph, so if that is already chordal then it
        will return that.

        Algorithm from Cano & Moral 1990 ->
        'Heuristic Algorithms for the Triangulation of Graphs'


        Parameters
        ----------
        *v* : a list (optional)
            A vertex list

        *e* : a list (optional)
            An edge list


        Returns
        -------
        *G* : a network Digraph object

        Effects
        -------
        None

        Notes
        -----
        Where is this used? Do we need to use networkx?       
        
        
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

        G = nx.Graph()
        G.add_edges_from(chordal_E)
        return G

    def is_chordal(self, edge_list=None):
        """
        Check if the graph is chordal/triangulated.

        Parameters
        ----------
        *edge_list* : a list of lists (optional)
            The edges to check (if not self.E)

        Returns
        -------
        *nx.is_chordal(G)* : a boolean
            Whether the graph is chordal or not

        Effects
        -------
        None

        Notes
        -----
        Again, do we need networkx for this? Eventually should
        write this check on our own.

        """
        if not edge_list:
            edge_list = self.E
        G = nx.Graph()
        G.add_edges_from(edge_list)
        return nx.is_chordal(G)
