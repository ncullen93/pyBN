"""
********
UnitTest
Clique
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(CliqueTestCase)
	return suite


class CliqueTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass