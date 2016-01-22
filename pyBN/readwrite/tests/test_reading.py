"""
********
UnitTest
Reading
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(ReadingTestCase)
	return suite


class ReadingTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass

	def test_read_bn_vertices:
		path = 'data/cmu.bn'
		bn = read_bn(path)
		self.assertListEqual(bn.V, ['Burglary','Earthquake','Alarm','JohnCalls','MaryCalls'])

	def test_read_bif_vertices:
		path = 'data/cancer.bif'
		bn = read_bn(path)
		self.assertListEqual(bn.V, ['Pollution', 'Smoker', 'Cancer', 'Xray', 'Dyspnoea'])