import os
import numpy as np
import pandas as pd

class DataLoader:
	"""Utility class to load data from files, given file directory"""
	def __init__(self, filedir):
		self.filedir = filedir
		
	def load(self):
		data = {}
		for split in ['train', 'test']:
			data[split] = []
			for sentiment in ['pos', 'neg']:
				encoder = 1 if sentiment == 'pos' else 0
				subdirpath = os.path.join(self.filedir, split, sentiment)
				filenames = os.listdir(subdirpath)
				for filename in filenames:
					filepath = os.path.join(subdirpath, filename)
					with open(filepath, "r") as review:
						data[split].append([review.read(), encoder])
		np.random.shuffle(data['train'])
		np.random.shuffle(data['test'])
		train_data = pd.DataFrame(data['train'], columns=['text', 'sentiment'])
		test_data = pd.DataFrame(data['test'], columns=['text', 'sentiment'])
		return train_data, test_data
