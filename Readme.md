# Bayesian Networks in Python

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

<h4>Parameter Learning</h4>
| Frequentist | Bayesian |
| ----------- | -------- |
| Maximum Likelihood Estimation | Dirichlet-Multinomial Estimation |


<h4>Classification</h4>
| Constrained (Tree) BNs  | General BNs
| ------------- | --------- |
| Naive Bayes    | MAP Inference Classification |
| Tree-Augmented Naive Bayes   |  |

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

<h2>Call for Researchers</h2>
If you are a neuroscience researcher or student who thinks the incredible learning/classification/inference functionality of Bayesian networks can add value to your work or want to see support for a specific type of data, please contact me and I will either a) answer questions or give advice on how to get the most out of pyBN or b) work with you to build custom functionality that integrates your work with the pyBN code. Email me at ncullen.th@dartmouth.edu.

<h2>References</h2>
- [1] Bielza, C., Larranaga, P. "Bayesian networks in neuroscience: a survey"
- [2] Daly, R., Shen, Q., Aitken, S. "Learning Bayesian networks: approaches and issues"
- [3] Bielza, C., Li, G., Larranaga, P. "Multi-dimensional classification with Bayesian networks"
- [4] Pham, D., Ruz, G. "Unsupervised training of Bayesian networks for data clustering"








