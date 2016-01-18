import numpy as np

class FastFactor(object):

    def __init__(self, bn, var):
        """
        A FastFactor uses a flattened numpy array for the cpt
        rather than a Pandas DataFrame. By storing the cpt in
        this manner and taking advantage of efficient algorithms,
        significant speedups occur.

        See Koller p.359

        Attributes:
            1. *bn*
            2. *var*
            3. *scope*
            5. *card*
            6. *stride*
            7. *cpt*

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

        if len(self.scope) == 1:
            self.cpt = np.array(bn.data[var]['cprob'])
        else:
            self.cpt = np.array([item for sublist in bn.data[var]['cprob'] for item in sublist])
        


    def multiply_factor(self, other_factor):
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
        ## I think it is only necessary when phi2 has vars not in phi1 ##
        rv_order.extend(list(set(phi2.scope).difference(set(phi1.scope))))
        phi1.card.update(phi2.card)
        #################################################################

        j,k=0,0
        assignment = dict([(rv, 0) for rv in scope_set])
        psi = np.zeros(np.product(phi1.card.values()))

        for i in range(len(psi)):
            psi[i] = phi1.cpt[j]*phi2.cpt[k]
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
        This function sums over one *rv* by keeping it constant.

        Thus, you end up with a factor whose scope is ONLY *rv*
        and whose length = cardinality of rv. 

        This is equivalent to calling self.sumout_var() over
        EVERY other variable in the scope and is thus faster
        when you want to do just that.
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

    def sumout_var_list(self, var_list):
        for var in var_list:
            self.sumout_var(var)

    def sumout_var(self, rv):
        """
        This function removes *rv* from the factor by summing
        over everything else.
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

    def maxout_var(self, rv):
        """
        This function removes *rv* from the factor by taking the 
        maximum value of all rv instantiations over everyting else.

        Used in MAP inference (i.e. Algorithm 13.1 in Koller p.557)
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
        Repeatly run self.reduce_factor() over a list of lists
        where a sublist = [rv,val]
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
        possibilities that don't align with [rv, val]

        Note, this will completely eliminate "rv" from the factor,
        including from the scope and cpt.
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
        self.cpt = np.log(self.cpt)
    def from_log(self):
        self.cpt = np.exp(self.cpt)

    def normalize(self):
        self.cpt = self.cpt / float(np.sum(self.cpt))