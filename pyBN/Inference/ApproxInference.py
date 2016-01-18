from pyBN.Models.ClusterGraph import *
from .Sampler import Sampler

class ApproxInference:
    """
    DEPRECATED - SHOULD BE SPLIT BY MAP/MARGINAL

    Attributes
    ----------


    Methods
    -------


    Notes
    -----

    """

    def __init__(self, BN):
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
        self.BN = BN

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
        sampler = Sampler(self.BN)
        sampler.forward_sample(n=n)

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
        sampler=Sampler(self.BN)
        sampler.gibbs_sample(n=n)

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
        sampler=Sampler(self.BN)
        sampler.likelihood_weighted_sample(target,evidence,n)




