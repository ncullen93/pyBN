# Bayesian Networks for Neuroscience

<h2>Overview</h2>
Many research fields have benefit from the flexibility and expressiveness of the Bayesian Network framework - medicine, economics, artificial intelligence, social science, and many more. Still, Bayesian networks have failed to penetrate the field of neuroscience, where they could be of incredible use. With that in mind, this project is aimed at making Bayesian Networks more accessible to neuroscientists. The motivation for this project is very much derived from the review paper by Bielza and Larranaga - "Bayesian networks in neuroscience: a survey."

As it relates to neuroscience, this code base aims to perform the following general tasks:
- <b><i>Function Connectivity Mapping</i></b>
	- looking for network-based relationships among variables of interest. This is
known in the more general case as "Association Discovery".
- <b><i>Feature Selection</i></b> 
	- choosing the appropriate features (variables) to use in classifying a target variable, or otherwise
reducing the dimensionality of a large dataset.
- <b><i>Supervised Classification</i></b>
	- predict class labels of new data from training data.
- <b><i>Unsupervised Classification</i></b>
	- learning from data where the ground truth is unknown. One example of this task is "clustering" - assigning data into clusters of similar observations.
- <b><i>Inference</i></b>
	- answering marginal or conditional probability queries, such as "what is the probability a person will develop dimentia given that he/she is 60 years old and had a stroke within the last five years?"

<h2>Data</h2>

The goal is to achieve full availability of state-of-the-art Bayesian network functionality for seamless use with all types of neuroscience data.
- Morphological Data
- Electrophysiological Data
- Genomics/Proteomics/Transcriptomics Data
- Neuroimaging Data
	- fMRI
	- MRI
	- EEG
- Others
	- ECoG, EMG, JME, TCD, DTI, PET

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








