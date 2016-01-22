<h1>Bayesian Networks in Python</h1>

<h2>Overview</h2>
This module provides a convenient and intuitive interface for reading/writing Discrete Bayesian Networks and performing fast inference over them. There seems to be a lack of many high-quality options for BNs in Python, so I hope this project will be a useful addition.

The package I have mostly used for BN's is "bnlearn" in the R language. It has incredible implementations of so many structure learning algorithms. However, its support and syntax for both Marginal and MAP inference is quite weak. This package takes care of that, at least.

I am a graduate student in the Di2Ag laboratory at Dartmouth College, and would love to collaborate on this project with anyone who has an interest in graphical models -- the class structure and syntax is easy to pick up! Shoot me an email at ncullen.th@dartmouth.edu

<h2>Usage</h2>
Getting up-and-running with this package is simple:

1. Click "Download ZIP" button towards the upper right corner of the page.
2. Unpack the ZIP file wherever you want on your local machine. You should now have a folder called "pyBN-master"
3. In your python terminal, change directories to be IN pyBN-master. Typing "ls" should show you "data", "examples" and "pyBN" folders. Stay in the "pyBN-master" directory for now!
4. In your python terminal, simply type "from pyBN import *". This will load all of the module's functions, classes, etc.
5. You are now free to use the package! Perhaps you want to start by creating a BayesNet object using "bn = BayesNet()" and so on.

<h2>Examples</h2>
This package includes a number of examples to help users get acquainted with the intuitive syntax and functionality of pyBN. For an updated list of examples, check out the collection of ipython notebooks in the "examples" folder located in the master directory.

Here is a list of current examples:
- ReadWrite : an introduction to reading (writing) BayesNet object from (to) files, along with an overview of the attributes and data structures inherit to BayesNet objects.


<h2>Current features:</h2>

<h4>Reading BNs</h4>
- .bif format
- json format

<h4>Drawing BNs</h4>
- Graphviz & networkx

<h4>Approx Inference</h4>
- loopy belief propagation
- forward sampling
- gibbs sampliing
- likelihood weighted sampling

<h4>Exact Inference</h4>
- Sum-Product Variable Elimination
- Max-Sum Variable Elimination
- Clique Tree Message Passing

<h4>Structure Learning</h4>
- Path Condition (PC) Algorithm
- Chow-Liu Tree Algorithm

<h4>Conditional Independence Tests</h4>
- Marginal Mutual Information (i.e. KL-Divergence)
- Conditional Mutual Information (i.e. Cross Entropy)
- Pearson X^2

<h2>Future Features:</h2>

<h4>Structure Learning</h4>
- Grow-shrink
- Hill-climbing

<h4>Parameter Learning</h4>
- Bayesian Estimation
- Maximum Likelihood Estimation
