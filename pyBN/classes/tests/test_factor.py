"""
********
UnitTest
Factor
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(FactorTestCase)
	return suite


class FactorTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass