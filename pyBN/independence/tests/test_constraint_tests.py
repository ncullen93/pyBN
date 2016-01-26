"""
****************
UnitTest
Constraint Tests
****************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np
import pandas as pd

class ConstraintTestsTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.data = np.loadtxt(os.path.join(self.dpath,'lizards.csv'),
			delimiter=',',
			dtype='int32',
			skiprows=1)

	def tearDown(self):
		pass

	def test_mi_two_vars(self):
		pass

	def test_mi_three_vars(self):
		pass

	def test_mi_four_vars(self):
		pass

	def test_chi2(self):
		pass