This module provides a convenient and intuitive interface for reading/writing Discrete Bayesian Networks and performing fast inference over them. There seems to be a lack of many high-quality options for BNs in Python, so I hope this project will be a useful addition.

The package I have mostly used for BN's is "bnlearn" in the R language. It has incredible implementations of so many structure learning algorithms. However, its support and syntax for both Marginal and MAP inference is quite weak. This package takes care of that, at least.

Current features:

Reading BNs:
- .bif format

Approx Inference:
- loopy belief propagation
- forward sampling
- gibbs sampliing
- likelihood weighted sampling

Exact Inference:
- Sum-Product Variable Elimination
- Clique Tree Message Passing/Belief Propagation


Future Features (To-Do List):

Structure Learning

Parameter Learning
- Bayesian Estimation
- Maximum Likelihood Estimation

