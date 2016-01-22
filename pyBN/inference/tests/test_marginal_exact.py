"""
**************
UnitTest
Marginal Exact
**************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(MarginalExactTestCase)
	return suite


class MarginalExactTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass