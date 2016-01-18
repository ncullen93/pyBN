from pyBN.Models.CliqueTree import *
from pyBN.Models.Factorization import *

class ExactInference:
    """
    DEPRECATED - SHOULD BE SPLIT BY MAP/MARGINAL

    Attributes
    ----------


    Methods
    -------


    Notes
    -----

    """


    def __init__(self, bn, operation=None):
        self.bn = bn
        self.operation = operation

    def sum_product_ve(self, target=None, evidence=None, order=None):
        factorization = Factorization(self.bn)
        factorization.marginal_ve(target,evidence,order)

    def clique_tree_bp(self, target=None, evidence=None, downward_pass=True):
        ctree = CliqueTree(self.bn)
        ctree.message_passing(target, evidence, downward_pass)

    def map(self, evidence=None, order=None):
        factorization = Factorization(self.bn)
        factorization.variable_elimination(marginal=False)