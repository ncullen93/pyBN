"""
********
UnitTest
Map Approx
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(MapApproxTestCase)
	return suite


class MapApproxTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass