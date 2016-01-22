"""
********************
GrowShrink Algorithm
********************

"Our approach constructs Bayesian networks by first identifying each node's
Markov blankets, then connecting nodes in a maximally consistent way. 
In contrast to the majority of work, which typically uses hill-climbing approaches 
that may produce dense and causally incorrect nets, our approach yields much more 
compact causal networks by heeding independenciesin the data. Compact causal networks
facilitate fast inference and are also easier to understand. We prove that under mild 
assumptions, our approach requires time polynomial in the size of the data and the number 
of nodes. A randomized variant, also presented here, yields comparable results at
much higher speeds" [1].

This algorithm relies on the "Markov Blanket", which is for a given random variable
the set of other random variables which render the given RV conditionally independent
from the rest of the network. That is, if you observe any of the variables in a given
RV's markov blanket, then observing the values of any OTHER variables in the network
will not change your beliefs about the given random variable.

The Markov blanket of a node X is easily identifiable from the graph: 
it consists of X's parents, X's children, and the parents of all of X's children.

The runtime of this algorithm is O(|V|) [1].

References
----------
[1] Margaritis and Thrun: "Bayesian Network Induction via Local Neighborhoods", 
NIPS 2000.

[2] https://www.cs.cmu.edu/~dmarg/Papers/PhD-Thesis-Margaritis.pdf

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.independence.constraint_tests import mi_test

def gs(data,
		alpha=0.05):
	"""
	Perform growshink algorithm over dataset to learn
	Bayesian network structure.

	STEPS
	-----
	1. Compute Markov Blanket
	2. Compute Graph Structure
	3. Orient Edges
	4. Remove Cycles
	5. Reverse Edges
	6. Propagate Directions

	Arguments
	---------
	*bn* : a BayesNet object
		BayesNet object you wish to modify
	*data* : pandas dataframe or nested numpy array
		Data from which you wish to learn structure

	Returns
	-------
	*bn* : a BayesNet object

	Effects
	-------
	None

	Notes
	-----

	"""
	n_rv = len(np.amax(data, axis=0))

	# STEP 1 : COMPUTE MARKOV BLANKETS
	B = dict([(rv,[]) for rv in bn.V])

	for X in range(n_rv):
		S = []

		grow_condition = True
		while grow_condition:

			grow_conditon=False
			for Y in range(n_rv):
				if X!=Y:
					# if there exists some Y such that Y is dependent on X given S,
					# add Y to S
					pval = mi_test(data[:,(X,Y,S)])
					if pval < alpha: # dependent
						grow_condition=True # dependent -> continue searching
						S.append(Y)

		shrink_condition = True
		while shrink_condition:

			shrink_condition=False
			for Y in S:
				S.remove(Y) # condition on S-{Y}
				# if X independent of Y given S-{Y}, leave Y out
				# if X dependent of Y given S-{Y}, keep it in
				pval = mi_test(data[:,(X,Y,S)])
				if pval < alpha: # dependent
					S.append(Y)
				else: # independent -> condition searching
					shrink_condition=True
		B[X] = S

	# STEP 2: COMPUTE GRAPH STRUCTURE
	edge_dict = dict([(rv,[]) for rv in range(n_rv)])
	for X in range(n_rv):
		for Y in B[X]:
			# X and Y are direct neighbors if X and Y are dependent
			# given S for all S in T, where T is the smaller of
			# B(X)-{Y} and B(Y)-{X}
			if len(B[X]) < len(B[Y]):
				T = copy(B[X])
				T.remove(Y)
			else:
				T = copy(B[Y])
				T.remove(X)

			direct_neighbors=True
			for S in T:
				pval = mi_test(data[:,(X,Y,S)])
				if pval > alpha:
					direct_neighbors=False
			if direct_neighbors:
				edge_dict[X].append(Y)
				edge_dict[Y].append(X)

	return edge_dict














