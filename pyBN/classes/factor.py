"""
************
Factor Class
************

This class holds a Conditional Probability Table structure
-- i.e. a factor. The benefit of this class structure is that
all factor manipulation happens in a centralized location,
thereby making it easier to write fast and readable code.

The Joint Probability Distribution of a Bayesian Network is
simply a product of its factors. Much of the functionality
is derived from algorithms presented in [1].

This class is a great candidate for using Numba JIT compilation
for manipulation and Numba's JIT functionality for the 
class itself.

References
----------
Koller, Friedman (2009). "Probabilistic Graphical Models."

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""


import numpy as np

class Factor(object):
    """
    A Factor uses a flattened numpy array for the cpt
    rather than a Pandas DataFrame. By storing the cpt in
    this manner and taking advantage of efficient algorithms,
    significant speedups occur.

    Attributes
    ----------

    *self.bn* : a BayesNet object

    *self.var* : a string
        The random variable to which this Factor belongs
    
    *self.scope* : a list
        The RV, and its parents (the RVs involved in the
        conditional probability table)
    
    *self.card* : a dictionary, where
        key = an RV in self.scope, and
        val = integer cardinality of the key (i.e. how
            many possible values it has)
    
    *self.stride* : a dictionary, where
        key = an RV in self.scope, and
        val = integer stride (i.e. how many rows in the 
            CPT until the NEXT value of RV is reached)
    
    *self.cpt* : a nested numpy array
        The probability values for self.var conditioned
        on its parents
    

    Methods
    -------
    *multiply_factor*
        Multiply two factors together. The factor
        multiplication algorithm used here is adapted
        from Koller and Friedman (PGMs) textbook.

    *sumover_var* :
        Sum over one *rv* by keeping it constant. Thus, you 
        end up with a 1-D factor whose scope is ONLY *rv*
        and whose length = cardinality of rv. 

    *sumout_var_list* :
        Remove a collection of rv's from the factor
        by summing out (i.e. calling sumout_var) over
        each rv.

    *sumout_var* :
        Remove passed-in *rv* from the factor by summing
        over everything else.

    *maxout_var* :
        Remove *rv* from the factor by taking the maximum value 
        of all rv instantiations over everyting else.

    *reduce_factor_by_list* :
        Reduce the factor by numerous sets of
        [rv,val]

    *reduce_factor* :
        Condition the factor by eliminating any sets of
        values that don't align with a given [rv, val]

    *to_log* :
        Convert probabilities to log space from
        normal space.

    *from_log* :
        Convert probabilities from log space to
        normal space.

    *normalize* :
        Make relevant collections of probabilities sum to one.


    Notes
    -----
    """
            


    def __init__(self, bn, var):
        """
        Initialize a Factor from a BayesNet object
        for a given random variable

        Arguments
        ---------
        *bn* : a BayesNet object

        *var* : a string
            The RV for which the Factor will be extracted.

        Effects
        -------
        - sets *self.bn*
        - sets *self.var*
        - sets *self.scope*
        - sets *self.card*
        - sets *self.stride*
        - sets *self.cpt*

        Notes
        -----

        """
        self.bn = bn
        self.var = var
        self.scope=[var]
        self.scope.extend(bn.data[var]['parents'])

        self.card = {} # key=rv, val=cardinality of the var
        for v in self.scope:
            self.card[v] = len(bn.data[v]['vals'])

        self.stride = {} # key=rv, val=stride of the var
        s=1
        for v in self.scope:
            self.stride[v]=s
            s*=self.card[v]

        assert (self.stride[self.var]==1), "Main Var should have stride = 1"

        if len(self.scope) == 1:
            self.cpt = np.array(bn.data[var]['cprob'])
        else:
            self.cpt = np.array([item for sublist in bn.data[var]['cprob'] for item in sublist])
        


    def multiply_factor(self, other_factor):
        """
        Multiply two factors together. The factor
        multiplication algorithm used here is adapted
        from Koller and Friedman (PGMs) textbook.

        In essence, the scope of the merged factor is the
        union of the two scopes.

        Arguments
        ---------
        *other_factor* : a different Factor object

        Returns
        -------
        None

        Effects
        -------
        - alters self.cpt
        - alters self.stride
        - alters self.card
        - alters self.scope

        Notes
        -----
        - What is done about normalization here? I guess
        assume it's already normalized

        """
        if len(self.scope)>=len(other_factor.scope):
            phi1=self
            phi2=other_factor
        else:
            phi1=other_factor
            phi2=self
        # go in order of strides to keep them in order after the fact
        rv_order = sorted(phi1.stride, key=phi1.stride.__getitem__)        
        scope_set=set(phi1.scope).union(set(phi2.scope))

        ########### HAD TO ADD THESE FOR V/E TO WORK... #################
        ## I think this is necessary when phi2 has vars not in phi1 ##
        rv_order.extend(list(set(phi2.scope).difference(set(phi1.scope))))
        phi1.card.update(phi2.card)
        #################################################################

        j,k=0,0
        assignment = dict([(rv, 0) for rv in scope_set])
        psi = np.zeros(np.product(phi1.card.values()))

        for i in range(len(psi)):
            psi[i] = round(phi1.cpt[j]*phi2.cpt[k],3)
            for rv in rv_order:
                assignment[rv] += 1
                if assignment[rv] == phi1.card[rv]:
                    assignment[rv] = 0
                    if rv in phi1.scope:
                        j=j-(phi1.card[rv]-1)*phi1.stride[rv]
                    if rv in phi2.scope:
                        k=k-(phi2.card[rv]-1)*phi2.stride[rv]
                else:
                    if rv in phi1.scope:
                        j=j+phi1.stride[rv]
                    if rv in phi2.scope:
                        k=k+phi2.stride[rv]
                    break
        self.cpt = psi
        self.scope = list(scope_set)
        self.card=phi1.card
        self.stride={}
        s=1
        for v in rv_order:
            self.stride[v]=s
            s*=phi1.card[v]


    def sumover_var(self, rv):
        """
        Sum over one *rv* by keeping it constant. Thus, you 
        end up with a factor whose scope is ONLY *rv*
        and whose length = cardinality of rv. 

        This is equivalent to calling self.sumout_var() over
        EVERY other variable in the scope and is thus faster
        when you want to do just that.

        Arguments
        ---------
        *rv* : a string
            The random variable to sum over.

        Returns
        -------
        None

        Effects
        -------
        - alters self.cpt
        - alters self.stride
        - alters self.card
        - alters self.scope

        Notes
        -----

        """
        exp_len = self.card[rv]
        new_cpt= np.zeros(exp_len)

        rv_card = self.card[rv]
        rv_stride = self.stride[rv]

        for i in range(exp_len):
            idx=i*rv_stride
            while idx < len(self.cpt):
                new_cpt[i] += np.sum(self.cpt[idx:(idx+rv_stride)])
                idx+=rv_card*rv_stride

        self.cpt=new_cpt
        self.card = {rv:rv_card}
        self.stride = {rv:1}
        self.scope = [rv]
        self.var = rv

        self.normalize()

    def sumout_var_list(self, var_list):
        """
        Remove a collection of rv's from the factor
        by summing out (i.e. calling sumout_var) over
        each rv.

        Arguments
        ---------
        *var_list* : a list
            The list of rv's to sum out.

        Returns
        -------
        None

        Effects
        -------
        - see "self.sumout_var"

        Notes
        -----

        """
        for var in var_list:
            self.sumout_var(var)

    def sumout_var(self, rv):
        """
        Remove passed-in *rv* from the factor by summing
        over everything else.

        Arguments
        ---------
        *rv* : a string
            The random variable to sum out

        Returns
        -------
        None

        Effects
        -------
        - alters self.cpt
        - alters self.stride
        - alters self.card
        - alters self.scope

        Notes
        -----     
        
        """
        exp_len = len(self.cpt)/self.card[rv]
        new_cpt = np.zeros(exp_len)
        
        rv_card = self.card[rv]
        rv_stride = self.stride[rv]

        k=0
        p = np.prod([self.card[r] for r in self.scope if self.stride[r] < self.stride[rv]])
        for i in range(exp_len):
            for c in range(rv_card):
                new_cpt[i] += self.cpt[k + (rv_stride*c)]
            k+=1
            if (k % p== 0):
                k += (p * (rv_card - 1))
        self.cpt=new_cpt

        del self.card[rv]
        self.stride.update((k,v/rv_card) for k,v in self.stride.items() if v > rv_stride)
        del self.stride[rv]
        self.scope.remove(rv)

        #self.normalize()

    def maxout_var(self, rv):
        """
        Remove *rv* from the factor by taking the maximum value 
        of all instantiations of the passed-in rv

        Used in MAP inference (i.e. Algorithm 13.1 in Koller p.557)

        Arguments
        ---------
        *rv* : a string
            The random variable

        Returns
        -------
        None

        Effects
        -------
        - alters self.cpt
        - alters self.stride
        - alters self.card
        - alters self.scope

        Notes
        -----        
        
        """
        self.cpt += 0.00002
        exp_len = len(self.cpt)/self.card[rv]
        new_cpt = np.zeros(exp_len)

        rv_card = self.card[rv]
        rv_stride = self.stride[rv]

        k=0
        p = np.prod([self.card[r] for r in self.scope if self.stride[r] < self.stride[rv]])
        for i in range(exp_len):
            max_val = 0
            for c in range(rv_card):
                if self.cpt[k + (rv_stride*c)] > max_val:
                    max_val = self.cpt[k + (rv_stride*c)]
                    new_cpt[i] = max_val
            k+=1
            if (k % p == 0):
                k += (p* (rv_card - 1))
        self.cpt=new_cpt

        del self.card[rv]
        self.stride.update((k,v/rv_card) for k,v in self.stride.items() if v > rv_stride)
        del self.stride[rv]
        self.scope.remove(rv)

    def reduce_factor_by_list(self, evidence):
        """
        Reduce the factor by numerous sets of
        [rv,val] -- this is done by running
        self.reduce_factor over the list of
        lists (*evidence*)

        Arguments
        ---------
        *evidence* : a list of lists/tuples
            The collection of rv-val pairs to
            remove from (condition upon) the factor


        Returns
        -------
        None

        Effects
        -------
        - see "self.reduce_factor"

        Notes
        -----
        - Again, might be good to check that each
            rv-val pair is actually in the factor
        """
        if isinstance(evidence, list):
            for rv,val in evidence:
                self.reduce_factor(rv,val)
        elif isinstance(evidence, dict):
            for rv,val in evidence.items():
                self.reduce_factor(rv,val)

    def reduce_factor(self, rv, val):
        """
        Condition the factor over evidence by eliminating any
        sets of values that don't align with [rv, val]

        Note, this will completely eliminate "rv" from the factor,
        including from the scope and cpt.

        Arguments
        ---------
        *rv* : a string
            The random variable to eliminate/condition upon.

        *val* : a string
            The value of RV

        Returns
        -------
        None

        Effects
        -------
        - alters self.cpt
        - alters self.scope
        - alters self.card
        - alters self.stride

        Notes
        -----
        - There are no fail-safes here to make sure the
            rv-val pair is actually in the factor..

        """
        exp_len = len(self.cpt)/float(self.card[rv])
        new_cpt=[]

        val_idx = self.bn.data[rv]['vals'].index(val)
        rv_card = self.card[rv]
        rv_stride = self.stride[rv]

        idx = val_idx*rv_stride
        while len(new_cpt) < exp_len:
            new_cpt.extend(self.cpt[idx:(idx+rv_stride)])
            idx+=rv_card*rv_stride

        self.cpt=new_cpt
        del self.card[rv]
        self.stride.update((k,v/rv_card) for k,v in self.stride.items() if v > rv_stride)
        del self.stride[rv]
        self.scope.remove(rv)

    def to_log(self):
        """
        Convert probabilities to log space from
        normal space.

        """
        self.cpt = np.log(self.cpt)

    def from_log(self):
        """
        Convert probabilities from log space to
        normal space.

        """
        self.cpt = np.exp(self.cpt)

    def normalize(self):
        """
        Make relevant collections of probabilities sum to one.

        This function is ALWAYS going to normalize the variable
        for which the stride = 1, because it's assumed that's the
        main/child variable.

        Effects
        -------
        - alters self.cpt

        Notes
        -----

        """
        var = [k for k,v in self.stride.items() if v==1][0]
        
        for i in range(0,len(self.cpt),self.card[var]):
            temp_sum = float(np.sum(self.cpt[i:(i+self.card[var])]))
            for j in range(self.card[var]):
                self.cpt[i+j] /= temp_sum
                self.cpt[i+j] = round(self.cpt[i+j],5)




