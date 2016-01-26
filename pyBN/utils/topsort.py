"""
Topological sort from a Factorization Object
"""
import pyBN.classes.factorization import Factorization

def topsort_from_factorization(F):
        """
        Return list of nodes in topological sort order.
        """
        assert (isinstance(F, Factorization)), 'F must be Factorization object'

        queue = [rv for rv in F.nodes() if F.num_parents(rv)==0]
        visited = []
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.append(vertex)
                for node in F.factor_dict.keys():
                    if vertex in F.factor_dict[node].parents():
                        queue.append(node)
        return visited