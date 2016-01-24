"""
********
UnitTest
ChowLiu
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(ChowLiuTestCase)
	return suite


class ChowLiuTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass