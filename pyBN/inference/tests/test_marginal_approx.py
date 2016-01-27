"""
***************
UnitTest
Marginal Approx
***************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.inference.marginal_approx import marginal_fs_a
from pyBN.readwrite.read import read_bn

class MarginalApproxTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.dpath,'cmu.bn'))

	def tearDown(self):
		pass

	def test_forward_sample(self):
		np.random.seed(3636)
		self.assertDictEqual(marginal_fs_a(self.bn,n=10000),
			{'Alarm': {'No': 0.997, 'Yes': 0.003},
			 'Burglary': {'No': 0.9987, 'Yes': 0.0013},
			 'Earthquake': {'No': 0.9981, 'Yes': 0.0019},
			 'JohnCalls': {'No': 0.943, 'Yes': 0.057},
			 'MaryCalls': {'No': 0.9889, 'Yes': 0.0111}}
			 )

	def test_likelihood_weighted_sample(self):
		np.random.seed(3636)
		self.assertDictEqual(marginal_lws_a(bn,evidence={'Burglary':'Yes'}),
			{'Alarm': {'No': 0.73, 'Yes': 0.27},
			 'Burglary': {'No': 0.0, 'Yes': 1.0},
			 'Earthquake': {'No': 0.997, 'Yes': 0.003},
			 'JohnCalls': {'No': 0.724, 'Yes': 0.276},
			 'MaryCalls': {'No': 0.802, 'Yes': 0.198}})

	def test_lws_allevidence(self):
		self.assertDictEqual(marginal_lws_a(bn,evidence={'Burglary':'Yes','Alarm':'Yes',
			'Earthquake':'Yes','JohnCalls':'Yes','MaryCalls':'Yes'}),
			{'Alarm': {'No': 0.0, 'Yes': 1.0},
			 'Burglary': {'No': 0.0, 'Yes': 1.0},
			 'Earthquake': {'No': 0.0, 'Yes': 1.0},
			 'JohnCalls': {'No': 0.0, 'Yes': 1.0},
			 'MaryCalls': {'No': 0.0, 'Yes': 1.0}})

	def test_gibbs(self):
		np.random.seed(3636)
		self.assertDictEqual(marginal_gs_a(bn,n=1000,burn=200),
			{'Alarm': {'No': 0.9988, 'Yes': 0.0},
				 'Burglary': {'No': 0.9988, 'Yes': 0.0},
				 'Earthquake': {'No': 0.9975, 'Yes': 0.0013},
				 'JohnCalls': {'No': 0.9313, 'Yes': 0.0675},
				 'MaryCalls': {'No': 0.9838, 'Yes': 0.015}})










