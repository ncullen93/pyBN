"""
********
UnitTest
BayesNet
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(BayesNetTestCase)
	return suite


class BayesNetTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass