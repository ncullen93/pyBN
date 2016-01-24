"""
********
UnitTest
OrientEdges
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(OrientEdgesTestCase)
	return suite


class OrientEdgesTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass