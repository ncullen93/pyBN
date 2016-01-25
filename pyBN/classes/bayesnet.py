"""
**************
BayesNet Class
**************

This is a class for creating/manipulating Bayesian Networks.
Currently, we only support Discrete Bayesian Networks.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



import numpy as np
import networkx as nx
import pandas as pd
from itertools import product
import copy
import numba
import time
import pdb


class BayesNet(object):
    """
    Overarching class for Bayesian Networks


    Attributes
    ----------

    *factors* : a dictionary where key = rv, value =  Factor() object

        Factor structure:
        
                *self.var* : a string
                    The random variable to which this Factor belongs
                
                *self.scope* : a list
                    The RV, and its parents (the RVs involved in the
                    conditional probability table)
                
                *self.stride* : a dictionary, where
                    key = an RV in self.scope, and
                    val = integer stride (i.e. how many rows in the 
                        CPT until the NEXT value of RV is reached)
                
                *self.cpt* : a 1D numpy array
                    The probability values for self.var conditioned
                    on its parents

                *self.vals* : a dictionary,
                    where key = an rv in self.scope and
                    value = a list of the values the rv can take

    *V* : a list of strings
        The Random Variables of the graph in topological sort order

    *E* : a list of tuples
        The edges of the graph in topological sort order

    *vals* : a dictionary where key = rv, value = list of rv's possible values

    GETTER Methods
    --------------


    SETTER Methods
    --------------


    
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

    def __init__(self,factors=None,vals=None):
        """
        Initialize the BayesNet class.

        Note that if this class is intialized w/ *factors* argument,
        self.V and self.E will be topsorted.

        Arguments
        ----------
        *factors* : a dictionary (OPTIONAL)
        *vals* : a dictionary (OPTIONAL)

        Notes
        -----
        
        """

        if factors:
            self.factors = factors
            self.V = self.topsort_nodes() # nodes are top sorted
            self.E = [(rv1,rv2) for rv1 in self.nodes() for rv2 in self.children(rv1)] # edges are topsorted
            self.vals = vals
        else:
            self.factors = dict
            self.V = []
            self.E = []
            self.vals = dict


    def __getitem__(self, rv):
        """
        Return the factor of a given Random Variable

        Arguments
        ---------
        *rv* : a string
            The random variable
        """
        return self.factors[rv]

    def __iter__(self):
        """
        An iterator over the BayesNet object,
        where each iteration yields a tuple with the
        first element being the random variable name and 
        the second element being its Factor object

        for i,j in bn:
            print i
            print j.cpt
        """
        return iter(self.factors.items())

    def __contains__(self, rv):
        """
        Boolean - whether a given Random Variable
        exists in the Bayesian network

        Arguments
        ---------
        *rv* : a string
            The random variable to check.
        """
        return rv in self.V

    def __len__(self):
        """
        The number of nodes in the graph
        """
        return len(self.nodes)

    def __str__(self):
        """
        What's printed to the console when the user
        types "print bn"
        """
        s = 'Conditional Dependencies \n'
        s +='------------------------\n'
        for rv in self.nodes():
            s += str(rv)
            if len(self.data[rv]['parents']) > 0:
                s += ' | '
                s += ','.join(self.data[rv]['parents'])
            else:
                s += ' (Prior)'
            s += '\n'
        return s

    def __repr__(self):
        """
        The representation of the BayesNet object:
        what's printed to the screen when the user
        types only "bn". For "print bn" see __str__.
        """
        s = '\n Conditional Dependencies \n'
        s +=' ------------------------\n'
        for rv in self.nodes():
            s += ' ' + str(rv)
            if len(list(self.parents(rv))) > 0:
                s += ' | '
                s += ','.join(list(self.parents(rv)))
            else:
                s += ' (Prior)'
            s += '\n'
        return s

    def as_dict(self):
        """
        Convert BayesNet object to a pure dictionary
        where key = rv and value = rv's factor which
        has also been converted into a dictionary.

        This is essentially how the BayesNet object
        gets written to file.

        Format
        ------
        'RV_NAME': {
                    'cpt' : list
                    'stride' : dictionary
                    'vals' : dictionary
        }

        Notes
        -----

        """
        bn_dict = dict((v,self.factors[v].as_dict()) for v in self.nodes())
        return bn_dict

    def topsort_nodes(self):
        """
        Return list of nodes in topological sort order.
        """
        queue = [rv for rv in self.factors.keys() if self.num_parents(rv)==0]
        visited = []
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.append(vertex)
                for node in self.factors.keys():
                    if vertex in self.factors[node].parents():
                        queue.append(node)
        return visited

    #### STRUCTURE ITERATORS ####

    def nodes(self):
        """
        Generator over nodes
        """
        for node in self.V:
            yield node

    def edges(self,topdown=True):
        """
        Generator over edges as tuples
        """
        if topdown:
            for e in self.E:
                yield e
        else:
            for e in reversed(self.E):
                yield e

    def parents(self, rv):
        """
        Return iterator
        """
        for parent in self.factors[rv].parents():
            yield parent

    def num_parents(self, rv):
        count = 0
        for parent in self.parents(rv):
            count +=1
        return count

    def children(self, rv):
        """
        Generator/iterator
        """
        for node in self.nodes():
            if rv in self.factors[node].parents():
                yield node

    def num_children(self, rv):
        count = 0
        for child in self.children(rv):
            count+=1
        return count

    def num_nodes(self):
        """
        The number of nodes in the graph
        """
        return len(self.V)

    ######
    def node_idx(self, rv):
        """
        Return integer index of a node string
        from self.V - this is a topological sort
        index as well.

        Example
        -------
        if V = ['X','Y','Z']
        then bn.node_idx('Y') = 1
        """
        return self.V.index(rv)

    def value_idx(self, rv, val):
        """
        Return integer index of a value of
        a given node from self.data[rv]['vals']
        """
        return self.vals[rv].index(val)

    def values(self, rv):
        """
        Generator over values of RV
        """
        for val in self.vals[rv]:
            yield val

    def card(self, rv):
        """
        Get the cardinality of variable *rv*
        """
        return len(self.vals[rv])

    def stride(self, rv, n):
        """
        Get the stride of a variable "n" IN the
        factor of variable "rv".
        """
        return self.factors[rv].stride[n]

    def scope_size(self, rv):
        """
        Return number of variables in the
        scope of rv's Factor
        """
        return len(self.factors[rv].scope)



    ###################### UTILITY METHODS ##############################

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
            Ex: {0:[1,2],1:[3],2:[],3:[]} -> should NOT have repeat edges

        *card_dict* : a dictionary where key = node, value = its cardinality

        Returns
        -------
        None

        Effects
        -------
        - Sets the structure information (self.data, self.E, self.V)

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
        self.data = dict((rv,
                    {'numoutcomes':card_dict[rv],
                    'vals':range(card_dict[rv]),
                    'parents':[i for i in edge_dict.keys() if rv in edge_dict[i]],
                    'children':edge_dict[rv],
                    'cprob':[]}) \
                        for rv in edge_dict.keys())        
        
        #self.E = [(str(i),str(j)) for i in edge_dict.keys() for j in edge_dict[i]]
        self.V = [str(v) for v in range(len(card_dict))]


    ###################### UTILITY METHODS ##############################
    def get_adj_list(self):
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
        edge_list = list(self.edges())
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
        e_list = copy.copy(list(self.edges()))
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

        # use set_structure instead ?
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
            edge_list = list(self.edges())
        G = nx.Graph()
        G.add_edges_from(edge_list)
        return nx.is_chordal(G)
