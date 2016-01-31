"""
********
UnitTest
Drawing
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(DrawingTestCase)
	return suite


class DrawingTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass