"""
********
UnitTest
BayesNet
********

Method	Checks that
assertEqual(a, b)	a == b	 
assertNotEqual(a, b)	a != b	 
assertTrue(x)	bool(x) is True	 
assertFalse(x)	bool(x) is False	 
assertIs(a, b)	a is b
assertIsNot(a, b)	a is not b
assertIsNone(x)	x is None
assertIsNotNone(x)	x is not None
assertIn(a, b)	a in b
assertNotIn(a, b)	a not in b
assertIsInstance(a, b)	isinstance(a, b)
assertNotIsInstance(a, b)	not isinstance(a, b)

assertAlmostEqual(a, b)	round(a-b, 7) == 0	 
assertNotAlmostEqual(a, b)	round(a-b, 7) != 0	 
assertGreater(a, b)	a > b
assertGreaterEqual(a, b)	a >= b
assertLess(a, b)	a < b
assertLessEqual(a, b)	a <= b

assertListEqual(a, b)	lists
assertTupleEqual(a, b)	tuples
assertSetEqual(a, b)	sets or frozensets
assertDictEqual(a, b)	dicts

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest

from pyBN.classes.bayesnet import BayesNet
from pyBN.readwrite.read import read_bn

import os
from os.path import dirname


class BayesNetTestCase(unittest.TestCase):

	def setUp(self):
		self.bn = BayesNet()
		path = dirname(dirname(dirname(dirname(__file__))))
		self.path = os.path.join(path,'data/cancer.bif')
		

	def tearDown(self):
		pass

	def test_isinstance(self):
		self.assertIsInstance(self.bn,BayesNet)

	def test_set_structure(self):
		edge_dict = {0:[1,2],1:[3],2:[],3:[]}
		card_dict = {0:2,1:2,2:2,3:2}
		self.bn.set_structure(edge_dict,card_dict)

		self.assertListEqual(self.bn.V,[0,1,2,3])
		self.assertListEqual(self.bn.E,[(0,1),(0,2),(1,3)])

	def test_get_adj_list(self):
		bn = read_bn(self.path)
		adj_list = bn.get_adj_list()
		self.assertListEqual(adj_list,[[2],[2],[3,4],[],[]])



if __name__ == '__main__':
	unittest.main(exit=False)




