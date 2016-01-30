"""
Lambda IAMB Code
"""



def lamba_iamb(data, alpha=0.05):
	"""
	Lambda IAMB Algorithm for learning the structure of a
	Discrete Bayesian Network from data. This Algorithm
	is similar to the iamb algorithm, except that it allows
	for a "lambda" coefficient that helps avoid false positives.

	This algorithm was originally developed for use as a
	feature selection algorithm - discovering the markov
	blanket of a target variable is equivalent to discovering
	the relevant features for classifications.

	In practice, this algorithm does just as well as a feature
	selection method compared to IAMB when naive bayes was 
	used as a classifier, but Lambda-iamb actually does much
	better than traditional iamb when traditional iamb does
	very poorly due to high false positive rates.

	Arguments
	---------
	*data* : a nested numpy array

	*dag* : a boolean
		Whether to return a Directed graph

	*pdag* : a boolean
		Whether to return a Partially Directed Graph.

	*alpha* : a float
		The type II error rate.

	Returns
	-------
	*bn* : a BayesNet object

	Effects
	-------
	None

	Notes
	-----
	"""
	Mb = dict([(rv,{}) for rv in range(n_rv)])
	n_rv = data.shape[1]

	# LEARN MARKOV BLANKET
	for T in xrange(n_rv):

		V = set(range(n_rv)) - {T}
		Mb_change=True

		# GROWING PHASE
		while Mb_change:
			Mb_change = False
			# find X_max in V-Mb(T)-{T} that maximizes 
			# mutual information of X,T|Mb(T)
			# i.e. max of mi_test(data[:,(X,T,Mb(T))])
			max_val = -1
			max_x = None
			for X in V - Mb(T) - {T}:
				cols = (X,T)+tuple(Mb(T))
				mi_val = mi_test(data[:,cols])
				if mi_val > max_val:
					max_val = mi_val
					max_x = X
			# if Xmax is dependent on T given Mb(T)
			cols = (max_x,T) + tuple(Mb(T))
			if mi_test(data[:,cols]) < alpha:
				Mb(T).add(X)
				Mb_change = True

		# SHRINKING PHASE
		for X in Mb(T):
			# if x is indepdent of t given Mb(T) - {x}
			cols = (X,T) + tuple(Mb(T)-{X})
			if mi_test(data[:,cols]) > alpha:
				Mb(T).remove(X)

	# RESOLVE GRAPH STRUCTURE
	edge_dict = resolve_markov_blanket(Mb, data)

	# ORIENT EDGES
	oriented_edge_dict = orient_edges_Mb(edge_dict,Mb,data,alpha)

	# CREATE BAYESNET OBJECT
	value_dict = dict(zip(range(data.shape[1]),
		[list(np.unique(col)) for col in data.T]))
	bn=BayesNet(oriented_edge_dict,value_dict)

	return edge_dict