"""
********
UnitTest
GrowShrink
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(GrowShrinkTestCase)
	return suite


class GrowShrinkTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass