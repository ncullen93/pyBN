
NOTE: I wrote this code to go along with Daphne Koller's book and no longer
maintain the repository, although the code should be easily adaptable. 
If you have any questions, please email me at ncullen at seas dot upenn dot edu.

<h1>Bayesian Networks in Python</h1>

<h2>Overview</h2>
This module provides a convenient and intuitive interface for reading, writing, plotting, performing inference, parameter learning, structure learning, and classification over Discrete Bayesian Networks - along with some other utility functions. There seems to be a lack of many high-quality options for BNs in Python, so I hope this project will be a useful addition.

I am a graduate student in the Di2Ag laboratory at Dartmouth College, and would love to collaborate on this project with anyone who has an interest in graphical models - Send me an email at ncullen.th@dartmouth.edu. If you're a researcher or student and want to use this module, I am happy to give an overview of the code/functionality or answer any questions.

For an up-to-date list of issues, go to the "issues" tab in this repository. Below is an updated list of features, along with information on usage/examples:


<h2>Current features</h2>

<h4>Marginal Inference</h4>
- Exact Marginal Inference
	- Sum-Product Variable Elimination 
	- Clique Tree Message Passing
- Approximate Marginal Inference
	- Forward Sampling  
	- Likelihood Weighted Sampling
	- Gibbs (MCMC) Sampling
	- Loopy Belief Propagation

<h4>MAP/MPE Inference</h4>
- Exact MAP Inference
	- Max-Product Variable Elimination
	- Integer Linear Programming
- Approximate MAP Inference
	- LP Relaxation

<h4>Constraint-Based Structure Learning</h4>
- Algorithms
	- PC
	- Grow-Shrink
	- IAMB/Lambda-IAMB/Fast-IAMB
- Independence Tests
	- Marginal Mutual Information
	- Conditional Mutual Information
	- Pearson Chi-Square

<h4>Score-Based Structure Learning</h4>
- Algorithms
	- Greedy Hill Climbing
	- Tabu Search
	- Random Restarts
- Scoring Functions
	- BIC/AIC/MDL
	- BDe/BDeu/K2

<h4>Tree-Based Structure Learning</h4>
- Naive Bayes
- Tree-Augmented Naive Bayes
- Chow-Liu

<h4>Hybrid Structure Leanring</h4>
- MMPC
- MMHC

<h4>Exact Structure Learning</h4>
- GOBNILP Solver

<h4>Parameter Learning</h4>
- Maximum Likelihood Estimation
- Dirichlet-Multinomial Estimation

<h4>Classification</h4>
- Naive Bayes
- Tree-Augmented Naive Bayes
- General DAG

<h4>Multi-Dimensional Classification</h4>
- Empty/Tree/Polytree/Forest
- General DAG

<h4>Comparing Two Bayesian Networks</h4>
- Structure-Based Distance Metrics
	- Missing Edges
	- Extra Edges
	- Incorrect Edge Orientation
	- Hamming Distance
- Parameter-Based Distance Metrics
	- KL-Divergence and JS-Divergence
	- Manhattan and Euclidean
	- Hellinger
	- Minkowski

<h4>Utility Functionality</h4>
- Determine Class Equivalence
- Discretize continuous data 
- Orient a PDAG
- Generate random sample dataset from a BN
- Markov Blanket operations

I previously wrote a Python wrapper for the GOBNILP project - a state-of-the-art integer programming solver for Bayesian network structure learning that can find the EXACT Global Maximum of any score-based objective function. It also links to CPLEX for incredible speed.
The wrappers can be found in the "pyGOBN" project at www.github.com/ncullen93/pyGOBN. For an overview of GOBNILP or to see its
great benchmarks on even the most massive datasets, visit https://www.cs.york.ac.uk/aig/sw/gobnilp/.


<h2>Examples</h2>
This package includes a number of examples to help users get acquainted with the intuitive syntax and functionality of pyBN. For an updated list of examples, check out the collection of ipython notebooks in the "examples" folder located in the master directory.

Here is a list of current examples:
- ReadWrite : an introduction to reading (writing) BayesNet object from (to) files, along with an overview of the attributes and data structures inherit to BayesNet objects.
- Drawing : an introduction to the drawing/plotting capabilities of pyBN with both small and large Bayesian networks.
- FactorOperations : an introduction to the Factor class, an exploration of the numerous attributes belonging to a Factor in
pyBN, an overview of every Factor operation function at the users' hands, and a short discussion of what makes Factor operations
so fast and efficient in pyBN.

<h2>Usage</h2>
Getting up-and-running with this package is simple:

1. Click "Download ZIP" button towards the upper right corner of the page.
2. Unpack the ZIP file wherever you want on your local machine. You should now have a folder called "pyBN-master"
3. In your python terminal, change directories to be IN pyBN-master. Typing "ls" should show you "data", "examples" and "pyBN" folders. Stay in the "pyBN-master" directory for now!
4. In your python terminal, simply type "from pyBN import ". This will load all of the module's functions, classes, etc.
5. You are now free to use the package! Perhaps you want to start by creating a BayesNet object using "bn = BayesNet()" and so on.

<h4>Unit Tests</h4>
If you want to test the functionality to make sure it all works on your local machine, navigate to the pybn-master directory and run the following command from the normal command-line (NOT ipython console):
- python -m unittest discover

