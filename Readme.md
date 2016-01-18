<h1>Bayesian Networks in Python</h1>

<h2>Overview</h2>
This module provides a convenient and intuitive interface for reading/writing Discrete Bayesian Networks and performing fast inference over them. There seems to be a lack of many high-quality options for BNs in Python, so I hope this project will be a useful addition.

The package I have mostly used for BN's is "bnlearn" in the R language. It has incredible implementations of so many structure learning algorithms. However, its support and syntax for both Marginal and MAP inference is quite weak. This package takes care of that, at least.

I am a graduate student in the Di2Ag laboratory at Dartmouth College, and would love to collaborate on this project with anyone who has an interest in graphical models -- the class structure and syntax is easy to pick up! Shoot me an email at ncullen.th@dartmouth.edu

<h2>Current features:</h2>

<h3>Reading BNs:</h3>
- .bif format
- json format

<h3>Approx Inference:</h3>
- loopy belief propagation
- forward sampling
- gibbs sampliing
- likelihood weighted sampling

<h3>Exact Inference:</h3>
- Sum-Product Variable Elimination
- Clique Tree Message Passing/Belief Propagation


<h2>Future Features (To-Do List):</h2>

<h3>Structure Learning</h3>
- Grow-shrink
- Hill-climbing

<h3>Parameter Learning</h3>
- Bayesian Estimation
- Maximum Likelihood Estimation

