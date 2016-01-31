"""
************
Clique Class
************

This is a class for creating/manipulating Cliques.

See CliqueTree class

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



class Clique(object):
    """
    Clique Class
    """

    def __init__(self, scope):
        """
        Instantiate a clique object

        Arguments
        ---------
        *scope* : a python set
            The set of variables in the cliques scope,
            i.e. the main var and its parents


        """
        self.scope=scope
        self.factors=[]
        self.psi = None # Psi should never change
        self.belief = None
        self.messages_received = []
        self.is_ready = False

    def send_initial_message(self, other_clique):
        """
        Send the first message to another clique.


        Arguments
        ---------
        *other_clique* : a different Clique object

        """
        psi_copy = copy.copy(self.psi)
        sepset = self.sepset(other_clique)
        sumout_vars = self.scope.difference(sepset)
        psi_copy.sumout_var_list(list(sumout_vars))
        psi_copy.cpt = psi_copy.cpt.loc[:,[c for c in psi_copy.cpt.columns if 'Prob' not in c]]
        psi_copy.cpt[str('Prob-Val-' + str(np.random.randint(0,1000000)))] = 1
        #print 'Init Msg: \n', psi_copy.cpt
        other_clique.messages_received.append(psi_copy)

        self.belief = copy.copy(self.psi)

    def sepset(self, other_clique):
        """
        The sepset of two cliques is the set of
        variables in the intersection of the two
        cliques' scopes.

        Arguments
        ---------
        *other_clique* : a Clique object

        """
        return self.scope.intersection(other_clique.scope)

    def compute_psi(self):
        """
        Compute a new psi (cpt) in order to 
        set the clique's belief.

        """
        if len(self.factors) == 0:
            print 'No factors assigned to this clique!'
            return None

        if len(self.factors) == 1:
            self.psi = copy.copy(self.factors[0])
        else:
            self.psi = max(self.factors, key=lambda x: len(x.cpt.columns))
            self.psi.merge_multiply(self.factors)
            self.belief = copy.copy(self.psi)

    def send_message(self, parent):
        """
        Send a message to the parent clique.

        Arguments
        ---------
        *parent* : a string
            The parent to which the message will
            be sent.

        """
        # First generate Belief = Original_Psi * all received messages
        if len(self.messages_received) > 0:
            # if there are messages received, mutliply them in to psi first
            if not self.belief:
                self.belief = copy.copy(self.psi)
            self.belief.merge_multiply(self.messages_received)
        else:
            # if there are no messages received, simply move on with psi
            if not self.belief:
                self.belief = copy.copy(self.psi)
        # generate message as belief with Ci - Sij vars summed out
        vars_to_sumout = list(self.scope.difference(self.sepset(parent)))
        message_to_send = copy.copy(self.belief)
        message_to_send.sumout_var_list(vars_to_sumout)
        parent.messages_received.append(message_to_send)

    def marginalize_over(self, target):
        """
        Marginalize the cpt (belief) over a target variable.

        Arguments
        ---------
        *target* : a string
            The target random variable.

        """
        self.belief.sumout_var_list(list(self.scope.difference({target})))

    def collect_beliefs(self):
        """        
        A root node (i.e. one that doesn't send a message) must run collect_beliefs
        since they are only collected at send_message()

        Also, we collect beliefs at the end of loopy belief propagation (approx. inference)
        since the main algorithm is just sending messages for a while.
        """
        if len(self.messages_received) > 0:
            self.belief = copy.copy(self.psi)
            for msg in self.messages_received:
                self.belief.cpt.merge(msg.cpt)
                #self.belief.normalize()
            self.belief.multiply_factor()
            print self.belief.cpt
            #self.belief.merge_multiply(self.messages_received)
            #self.belief.normalize()
            self.messages_received = []
        else:
            self.belief = copy.copy(self.belief)
            #print 'No Messages Received - Belief is just original Psi'



            