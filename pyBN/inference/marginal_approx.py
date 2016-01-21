"""
******************************
Approximate Marginal Inference
******************************

Perform approx. marginal inference over a BayesNet object,
with or without evidence.

Eventually, there will be a wrapper function "marginal_approx"
for all of the algorithms, and users can choose their method as
an argument to that function.

Approximate Marginal Inference Algorithms
-----------------------------------------
	
	- Forward Sampling
	- Likelihood Weighted Sampling
	- Gibbs (MCMC) Sampling
	- Loopy Belief Propagation

References
----------
[1] Koller, Friedman (2009). "Probabilistic Graphical Models."

"""

__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""




def forward_sample(self, n=1000):
	"""
	Overview
	--------


	Parameters
	----------


	Returns
	-------


	Notes
	-----
	"""
	sample_dict = {}

	G = self.BN.get_networkx()
	rv_order = nx.topological_sort(G)

	#factor_dict = dict([(var,FastFactor(self.BN, var)) for var in self.BN.V])
	parent_dict = dict([(var, G.predecessors(var)) for var in self.BN.V])

	for var in self.BN.V:
	    sample_dict[var] = {}
	    for val in self.BN.data[var]['vals']:
	        sample_dict[var][val] = 0

	for i in range(n):
	    if i % (n/float(10)) == 0:
	        print 'Sample: ' , i
	    new_sample = {}
	    for rv in rv_order:
	        f = FastFactor(self.BN,rv)
	        for p in parent_dict[rv]:
	            f.reduce_factor(p,new_sample[p])
	        choice_vals = self.BN.data[rv]['vals']
	        choice_probs = f.cpt
	        chosen_val = np.random.choice(choice_vals, p=choice_probs)

	        sample_dict[rv][chosen_val] += 1
	        new_sample[rv] = chosen_val

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] = int(sample_dict[rv][val]) / float(n)
	self.forward_counter=sample_dict

def lw_sample(self, n=1000, evidence={}):
	"""
	Overview
	--------


	Parameters
	----------


	Returns
	-------


	Notes
	-----
	"""
	sample_dict = {}
	weight_list = np.ones(n)

	G = self.BN.get_networkx()
	rv_order = nx.topological_sort(G)

	#factor_dict = dict([(var,FastFactor(self.BN, var)) for var in self.BN.V])
	parent_dict = dict([(var, G.predecessors(var)) for var in self.BN.V])

	for var in self.BN.V:
	    sample_dict[var] = {}
	    for val in self.BN.data[var]['vals']:
	        sample_dict[var][val] = 0

	for i in range(n):
	    if i % (n/float(10)) == 0:
	        print 'Sample: ' , i
	    new_sample = {}
	    for rv in rv_order:
	        f = FastFactor(self.BN,rv)
	        # reduce_factor by parent samples
	        for p in parent_dict[rv]:
	            f.reduce_factor(p,new_sample[p])
	        # if rv in evidence, choose that value and weight
	        if rv in evidence.keys():
	            chosen_val = evidence[rv]
	            weight_list[i] *= f.cpt[self.BN.data[rv]['vals'].index(evidence[rv])]
	        # if rv not in evidence, sample as usual
	        else:
	            choice_vals = self.BN.data[rv]['vals']
	            choice_probs = f.cpt
	            chosen_val = np.random.choice(choice_vals, p=choice_probs)
	            
	        new_sample[rv] = chosen_val
	    # weight the choice by the evidence likelihood    
	    for rv in new_sample:
	        sample_dict[rv][new_sample[rv]] += 1*weight_list[i]

	weight_sum = sum(weight_list)

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] /= weight_sum
	        sample_dict[rv][val] = round(sample_dict[rv][val],4)
	self.lw_counter=sample_dict


def gibbs_sample(self, n=1000):
	"""
	Overview
	--------


	Parameters
	----------


	Returns
	-------


	Notes
	-----
	"""
	G=self.BN.get_networkx()
	bn=self.BN
	counter={}
	for rv in bn.V:
	    counter[rv]={}
	    for val in bn.data[rv]['vals']:
	        counter[rv][val] = 0

	state = {}
	for rv in bn.V:
	    state[rv] = np.random.choice(bn.data[rv]['vals']) # uniform sample

	for i in range(n):
	    if i % (n/float(10)) == 0:
	        print 'Sample: ' , i
	    for rv in bn.V:
	        # get possible values conditioned on everything else
	        parents = G.predecessors(rv)
	        # no parents - prior
	        if len(parents) == 0:
	            choice_vals = bn.data[rv]['vals']
	            choice_probs = bn.data[rv]['cprob']
	        # has parent - filter cpt
	        else:
	            f = FastFactor(bn,rv)
	            for p in parents:
	                f.reduce_factor(p,state[p])
	            choice_vals = bn.data[rv]['vals']
	            choice_probs = f.cpt
	        # sample over remaining possibilities
	        chosen_val = np.random.choice(choice_vals,p=choice_probs)
	        state[rv]=chosen_val
	    # update counter dictionary
	    if i > burn:
	        for rv,val in state.items():
	            counter[rv][val] +=1

	for rv in counter:
	    for val in counter[rv]:
	        counter[rv][val] = round(int(counter[rv][val]) / float(n-burn),4)
	self.gibbs_counter=counter


def loopy_bp(self, target=None, evidence=None):
	"""

	Overview
	--------


	Parameters
	----------


	Returns
	-------


	Notes
	-----

	"""
	cgraph = ClusterGraph(self.BN)
	cgraph.loopy_belief_propagation(target, evidence)