"""
************
UnitTest
ClusterGraph
************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(ClusterGraphTestCase)
	return suite


class ClusterGraphTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass