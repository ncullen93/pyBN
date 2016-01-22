"""
********
UnitTest
PC
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(PCTestCase)
	return suite


class PCTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass