<h1>Bayesian Networks in Python</h1>

<h2>Overview</h2>
This module provides a convenient and intuitive interface for reading, writing, plotting, performing inference, parameter learning, structure learning, and classification over Discrete Bayesian Networks - along with some other utility functions. There seems to be a lack of many high-quality options for BNs in Python, so I hope this project will be a useful addition.

I am a graduate student in the Di2Ag laboratory at Dartmouth College, and would love to collaborate on this project with anyone who has an interest in graphical models - Send me an email at ncullen.th@dartmouth.edu.

For an up-to-date list of issues, go to the "issues" tab in this repository. Below is an updated list of features, along with information on usage/examples:

<h2>Current features:</h2>

<h4>FileIO and Drawing</h4>
| Reading BNs   | Drawing BNs   |
| ------------- | ------------- | 
| BIF format    | Graphviz Engine  |
| JSON format   | Networkx      |


<h4>Inference</h4>
| Exact Marginal Inference  | Approximate Marginal Inference  | Exact MAP Inference |
| ------------- | ------------- | ----------------- |
| Sum-Product Variable Elimination   | Forward Sampling    | Max-Sum Variable Elimination   |
| Clique Tree Message Passing  | Likelihood Weighted Sampling     |     
|				|		Gibbs (MCMC) Sampling 		| 			|
|				|		Loopy Belief Propagation			| 		|

<h4>Structure Learning</h4>
| Constraint-Based  | Tree-Based | Independence Tests |
| ------------- | ------------- | ----------------- |
| Path Condition (PC) Algorithm   | Naive Bayes   | Marginal Mutual Information (KL-Divergence)    |
| Grow-Shrink (GS) Algorithm   | Tree-Augmented Naive Bayes    | Conditional Mutual Information (Cross Entropy)     |
| 	IAMB Algorithm		|		Chow-Liu Algorithm		| Pearsion Chi-Square|
| 	Lambda-IAMB Algorithm		|				| 			|
| 	Fast-IAMB Algorithm		|				|			|

<h4>Parameter Learning</h4>
| Frequentist | Bayesian |
| ----------- | -------- |
| Maximum Likelihood Estimation | Dirichlet-Multinomial Estimation |


<h4>Classification</h4>
| Constrained (Tree) BNs  | General BNs
| ------------- | --------- |
| Naive Bayes    | MAP Inference Classification |
| Tree-Augmented Naive Bayes   |  |

<h4>Distance/Model Comparison Metrics</h4>
| Structure-Based Distance Metrics   | Parameter-Based Distance Metrics  |
| ------------- | ------------- | 
| Missing Edges   | KL-Divergence and JS-Divergence|
| Extra Edges  | Manhattan and Euclidean |
| Incorrect Edge Orientation				|Hellinger				| 
|	Hamming Distance			|		Minkowski	| 

<h4>Other</h4>
|Utility|
| ----- |
|Discretize Continuous Data |
|Markov Blanket Feature Selection |

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

