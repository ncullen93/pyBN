"""
********
UnitTest
Misc
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(MiscTestCase)
	return suite


class MiscTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass