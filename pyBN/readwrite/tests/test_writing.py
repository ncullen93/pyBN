"""
********
UnitTest
Writing
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(WritingTestCase)
	return suite


class WritingTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass

	def test_write_bn():
		bn = read_bn('data/cancer.bif') # read a file
		write_bn('data/write_test.bn') # write it to file
		bn2 = read_bn('data/write_test.bn') # read it back

		assert_equal(bn.V,bn2.V)
		assert_equal(bn.E,bn2.E)
		assert_equal(bn.data,bn2.data)