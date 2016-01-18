import pandas as pd
import numpy as np

class StructureLearner(object):

	def __init__(self,
				data=None,
				dtype=None):
		"""
		Class for structure learning.

		Attributes
		----------

		data : a Pandas dataframe or numpy 3D array
			The data from which the user wants to learn

		dtype : 'pandas' | 'numpy'
			The type of data

		Methods
		-------

		Algorithms implemented in 'bnlearn':

			constraint-based structure learning algorithms:

				Grow-Shrink (GS);
				Incremental Association Markov Blanket (IAMB);
				Fast Incremental Association (Fast-IAMB);
				Interleaved Incremental Association (Inter-IAMB);
			
			score-based structure learning algorithms:

				Hill Climbing (HC);
				Tabu Search (Tabu);
			
			hybrid structure learning algorithms:

				Max-Min Hill Climbing (MMHC);
					General 2-Phase Restricted Maximization (RSMAX2);
			
			local discovery algorithms:

				Chow-Liu;
				ARACNE;
				Max-Min Parents & Children (MMPC);
				Semi-Interleaved Hiton-PC (SI-HITON-PC);

		"""
		if is_instance(data, 'pd.DataFrame'):
			self.dtype = 'pandas'
		elif is_instance(data, 'np.ndarray'):
			self.dtype = 'numpy'

		self.data = data







