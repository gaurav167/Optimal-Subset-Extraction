from model import Model

class Indivisual:
	"""Class for indivisual member of the population"""
	
	def __init__(self, vocabulary, train_data, test_data):
		self.vocabulary = vocabulary
		self.fitness = 0
		self.train_data = train_data
		self.test_data = test_data
		self.evaluate()

	def getFitness(self):
		return self.fitness

	def getVocabulary(self):
		return self.vocabulary

	def setVocabulary(self, vocabulary):
		self.vocabulary = vocabulary
		self.evaluate()

	def evaluate(self):
		# Evaluate Fitness from Model
		self.model = Model(self.vocabulary, self.train_data, self.test_data)
		self.fitness = self.model.calcAccuracy()