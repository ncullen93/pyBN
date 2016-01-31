"""
****************
Bayesian Network
Classifiers
****************

Implementation of various Bayesian network
classifiers.

"""


def predict(data, target, classifier=None, method='nb'):
	"""
	Wrapper for a unified interface to the 
	various classification algorithms. The pyBN
	user can call any algorithm from this function for
	convienence.

	The prediction algorithm works as follows:
		- For each row of data, set the observed attribute
			variables as evidence
		- Pick the most likely value of the target variable
			by either a) running MAP inference and simply
			returning the chosen value, or b) running Marginal
			inference and selecting the value of the target variable
			with the highest probability.
		- Compare the chosen value to the actual value.
		- Return the actual values, the chosen values, and the accuracy score.

	Arguments
	---------
	*classifer* : a BayesNet object (optional)
		The BayesNet model to use as the classifier model. 
		NOTE: The user can supply a BayesNet class which has
		been learned as a general bn, naive bayes, tan, or ban model 
		from the appropriate structure learning algorithm or file...
		OR the user can leave classifier as "None" and supply a string
		to *method*, in which class the corresponding model will be
		structure/parameter learned IN THIS FUNCTION and used as the
		classifier.
		
	*data* : a nested numpy array

	*target* : an integer
		The data column corresponding to the
		predictor/target variable.

	*method* : a string
		Which BN classifier to use if *classifier* is None.
		Options:
			- 'bn' : general bayesian network classifier
			- 'nb' : naive bayes classifier
			- 'tan' : tree-augmented naive bayes
			- 'ban' : bayesian augmented naive bayes ?

	

	Returns
	-------
	*c_dict* : a dictionary, where
		keys = 'y', 'yp', and 'acc', where
		'y' is the true target values (np array),
		'yp' is the predicted target values (np array),
		'acc' is the prediction accuracy percentage (float).

	Effects
	-------
	None

	Notes
	-----
	"""
	if classifier is None:
		classifier = learn_structure(data, method)
		learn_parameters(classifier)


	### CLASSIFIER PREDICTION FROM INFERENCE ###
	for row in data:
		pass
			























