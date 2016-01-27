"""
**************
UnitTest
Marginal Exact
**************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.readwrite.read import read_bn
from pyBN.inference.marginal_exact import marginal_ve_e



class MarginalExactTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.dpath,'cmu.bn'))
		
	def tearDown(self):
		pass

	def test_marginal_ve_e_1(self):
		self.assertListEqual(list(marginal_ve_e(self.bn,target='JohnCalls')),
			[0.94781,0.05219])