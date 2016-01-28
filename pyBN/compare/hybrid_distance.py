"""
****************
BN Hybrid
Distance Metrics
****************

Code for computing the distance between 
2+ Bayesian networks that takes into account
both structure AND parameters. 

Any structure-based distance metric can be
combined with any parameter-based distance metric
to compute the hybrid distance.

Also, we support a type of supervised approach to
hybrid distance measurement where the relative importance
placed on the structure and distance metrics are varied
such that maximum (or minimum) closeness is achieved.

"""

def hybrid_distance(x,y, structure_metric, parameter_metric):
	pass