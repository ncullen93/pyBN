"""
***************
UnitTest
Marginal Approx
***************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(MarginalApproxTestCase)
	return suite


class MarginalApproxTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass