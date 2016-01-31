"""
****************
Bayesian Network
Classifiers
****************

Implementation of various Bayesian network
classifiers.

"""


def classify(data, target, method='bn'):
	"""
	Wrapper for a unified interface to the 
	various classification algorithms. The pyBN
	user can call any algorithm from this function for
	convienence.

	Arguments
	---------
	*data* : a nested numpy array

	*target* : an integer
		The data column corresponding to the
		predictor/target variable.

	*method* : a string
		Which BN classifier to use -
		Options:
			- 'bn' : general bayesian network classifier
			- 'nb' : naive bayes classifier
			- 'tan' : tree-augmented naive bayes
			- 'ban' : bayesian augmented naive bayes ?

	Returns
	-------
	*classify* : a dictionary, where
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
	pass



