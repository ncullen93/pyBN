"""
********
UnitTest
PC
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.structure_learn.path_condition import pc


class PCTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		path = (os.path.join(self.dpath,'lizards.csv'))
		self.data = np.loadtxt(path, dtype='int32',skiprows=1,delimiter=',')
	
	def tearDown(self):
		pass

	def test_pc1(self):
		