"""
********
UnitTest
Map Exact
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(MapExactTestCase)
	return suite


class MapExactTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass