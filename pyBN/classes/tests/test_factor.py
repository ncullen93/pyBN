"""
********
UnitTest
Factor
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
import os
from os.path import dirname
import numpy as np

from pyBN.classes.factor import Factor


class FactorTestCase(unittest.TestCase):

	def setUp(self):
		self.data_path = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.data_path,'cmu.bn'))
		self.f = Factor(self.bn, 'Alarm')
	def tearDown(self):
		pass

	# Factor Creation Tests
	def test_factor_init(self):
		assertIsInstance(self.f,Factor)

	def test_factor_bn(self):
		assertListEqual(self.f.bn.V,
			['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls'])

	def test_factor_var(self):
		assertEqual(self.f.var, 'Alarm')
	
	def test_factor_scope(self):
		assertListEqual(self.f.scope,['Alarm','Burglary','Earthquake'])
	
	def test_factor_card(self):
		assertDictEqual(self.f.card,
			{'Alarm':2, 'Burglary':2, 'Earthquake':2})
	
	def test_factor_stride(self):
		assertDictEqual(self.f.stride,
			{'Alarm':1, 'Burglary':2, 'Earthquake':4})
	
	def test_factor_cpt(self):
		assertListEqual(self.f.cpt,
			[ 0.999,  0.001,  0.71 ,  0.29 ,  0.06 ,  0.94 ,  0.05 ,  0.95 ])
	
	# Factor Operations Tests
	def test_multiply_factor(self):
		pass
	
	def test_sumover_var(self):
		pass

	def test_sumout_var_list(self):
		pass

	def test_sumout_var(self):
		pass

	def test_maxout_var(self):
		pass

	def test_reduce_factor_by_list(self):
		pass

	def test_reduce_factor(self):
		pass

	def test_to_log(self):
		pass

	def test_from_log(self):
		pass

	def test_normalize(self):
		pass
    
if __name__ == '__main__':
	unittest.main(exit=False)









