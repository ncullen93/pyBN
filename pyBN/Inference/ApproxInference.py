from pyBN.Models.ClusterGraph import *
from .Sampler import Sampler

class ApproxInference:

    def __init__(self, BN):
        self.BN = BN

    def loopy_bp(self, target=None, evidence=None):
        cgraph = ClusterGraph(self.BN)
        cgraph.loopy_belief_propagation(target, evidence)

    def forward_sample(self, n=1000):
        sampler = Sampler(self.BN)
        sampler.forward_sample(n=n)

    def gibbs_sample(self, n=1000):
        sampler=Sampler(self.BN)
        sampler.gibbs_sample(n=n)

    def lw_sample(self, n=1000, evidence={}):
        sampler=Sampler(self.BN)
        sampler.likelihood_weighted_sample(target,evidence,n)