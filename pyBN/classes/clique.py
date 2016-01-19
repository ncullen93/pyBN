"""
******************
Clique Class
******************

See CliqueTree class

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



class Clique(object):
    """
    Class for Cliques


    Attributes
    ----------


    Methods
    -------


    Notes
    -----

    """

    def __init__(self, scope):
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
        self.scope=scope
        self.factors=[]
        self.psi = None # Psi should never change
        self.belief = None
        self.messages_received = []
        self.is_ready = False

    def send_initial_message(self, other_clique):
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
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----


        """
        return self.scope.intersection(other_clique.scope)

    def compute_psi(self):
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
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----


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
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        """
        self.belief.sumout_var_list(list(self.scope.difference({target})))

    def collect_beliefs(self):
        """
        Overview
        --------


        Parameters
        ----------


        Returns
        -------


        Notes
        -----

        
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



            