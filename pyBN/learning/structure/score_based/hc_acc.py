"""
Hill Climbing Prediction Accuracy algorithm to add, delete, or reverse
arcs in a Multi-Dimensional Bayesian Network Classifier 
such that the move with the highest increase in classifier 
accuracy is chosen. If there is no move that increases 
the accuracy of the classifier, the algorithm terminates.

Clearly, this algorithm is different than traditional 
structure learning approaches because it tunes explicitly 
for classifier accuracy. By the same token, it is not 
clear the extent to which the wrapper algorithm encourages
or even causes overfitting and thus lack of classifier
generalization. That is an interesting question to research.

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""


def hc_pa(data):
	pass