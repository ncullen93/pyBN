

def test_read_bn_vertices:
	path = 'data/cmu.bn'
	bn = read_bn(path)
	assert_equal(bn.V, ['Burglary','Earthquake','Alarm','JohnCalls','MaryCalls'])

def test_read_bif_vertices:
	path = 'data/cancer.bif'
	bn = read_bn(path)
	assert_equal(bn.V, ['Pollution', 'Smoker', 'Cancer', 'Xray', 'Dyspnoea'])