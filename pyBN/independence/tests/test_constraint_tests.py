"""
****************
UnitTest
Constraint Tests
****************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(ConstraintTestsTestCase)
	return suite


class ConstraintTestsTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass