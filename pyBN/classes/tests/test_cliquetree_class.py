"""
**********
UnitTest
CliqueTree
**********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(CliqueTreeTestCase)
	return suite


class CliqueTreeTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass