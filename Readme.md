<h1>Bayesian Networks in Python</h1>

Note: This code base is essentially the same as the "neuroBN" package found in www.github.com/ncullen93/neuroBN. I maintain two separate repositories because I expect the two projects to diverge sharply in the near future.

<h2>NEW FEATURE!</h2>
I just wrote a Python wrapper for the GOBNILP project - a state-of-the-art integer programming solver for Bayesian network structure learning that can find the EXACT Global Maximum of any score-based objective function. It also links to CPLEX for incredible speed.
The wrappers can be found in the "pyGOBN" project at www.github.com/ncullen93/pyGOBN. For an overview of GOBNILP or to see its
great benchmarks on even the most massive datasets, visit https://www.cs.york.ac.uk/aig/sw/gobnilp/.

<h2>Overview</h2>
This module provides a convenient and intuitive interface for reading, writing, plotting, performing inference, parameter learning, structure learning, and classification over Discrete Bayesian Networks - along with some other utility functions. There seems to be a lack of many high-quality options for BNs in Python, so I hope this project will be a useful addition.

I am a graduate student in the Di2Ag laboratory at Dartmouth College, and would love to collaborate on this project with anyone who has an interest in graphical models - Send me an email at ncullen.th@dartmouth.edu. If you're a researcher or student and want to use this module for any reason, I am happy to give an overview of the code/functionality or answer any questions.

For an up-to-date list of issues, go to the "issues" tab in this repository. Below is an updated list of features, along with information on usage/examples:

<h2>Current features</h2>

<h4>FileIO and Drawing</h4>
| Reading BNs   | Drawing BNs   |
| ------------- | ------------- | 
| BIF format    | Graphviz Engine  |
| JSON format   | Networkx      |


<h4>Marginal Inference</h4>
| Exact Marginal Inference  | Approximate Marginal Inference  | 
| ------------- | ------------- | ----------------- |
| Sum-Product Variable Elimination   | Forward Sampling    |
| Clique Tree Message Passing  | Likelihood Weighted Sampling     |  
|				|		Gibbs (MCMC) Sampling 		|
|				|		Loopy Belief Propagation			| 

<h4>MAP Inference</h4>
| Exact MAP Inference | Approximate MAP Inference |
| ------------------- | ------------------------- |
| Max-Product Variable Elimination | LP Relaxation |
| Integer Linear Programming	|			|

<h4>Constraint-Based Structure Learning</h4>
| Algorithms  | Independence Tests |
| ------------- | ----------------- |
| Path Condition (PC)   | Marginal Mutual Information (KL-Divergence)    |
| Grow-Shrink (GS)  | Conditional Mutual Information (Cross Entropy)     |
| 	IAMB 	| Pearsion Chi-Square|
| 	Lambda-IAMB 	|				| 	
| 	Fast-IAMB 	|				|		

<h4>Score-Based Structure Learning</h4>
| Algorithms | Score Metrics |
| ---------- | ------------- |
| Hill Climbing | Log-Likelihood |
| Tabu List	| AIC/BIC	|
| Random Restarts | 		|

<h4>Tree-Based Structure Learning</h4>
| Algorithms |
| ---------- |
| Naive Bayes |
| Tree-Augmented Naive Bayes |
| Chow-Liu	|

<h4>Hybrid Structure Leanring</h4>
| Algorithms |
| --------- |
| MMPC		|
| MMHC		|

<h4>Exact Structure Learning</h4>
| Algorithms |
| ---------- |
| GOBNILP Solver |

<h4>Parameter Learning</h4>
| Frequentist | Bayesian |
| ----------- | -------- |
| Maximum Likelihood Estimation | Dirichlet-Multinomial Estimation |


<h4>Classification</h4>
| Constrained (Tree) BNs  | General BNs
| ------------- | --------- |
| Naive Bayes    | MAP Inference Classification |
| Tree-Augmented Naive Bayes   |  |

<h4>Multi-Dimensional Classification</h4>
| Class/Feature Subgraphs |
| ---------------------- |
| Empty |
| Tree/Polytree/Forest |
| General DAG |

<h4>Comparing Two Bayesian Networks</h4>
| Structure-Based Distance Metrics   | Parameter-Based Distance Metrics  | 
| ------------- | ------------- | 
| Missing Edges   | KL-Divergence and JS-Divergence|
| Extra Edges  | Manhattan and Euclidean |
| Incorrect Edge Orientation				|Hellinger				| 
|	Hamming Distance			|		Minkowski	| 

<h4>Utility Functionality</h4>
| BN Utility | Other |
| ---------- | ----- |
| 	Determine Class Equivalence| Discretize continuous data  |
| Orient a PDAG from Markov Blanket or Block Set | Generate random sample dataset from a BN |
| Elimination Ordering Heuristics | Topological Sort Algorithm |
| Get Markov Blanket of a BN | Minimum Spanning Tree Algorithm |
| Markov Blanket Fitness Metric |
| Make a Chordal or Moral BN |



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

