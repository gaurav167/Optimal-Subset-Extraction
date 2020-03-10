import random

import CONSTANTS


class Population:
	"""Class for population of genetic algorithm"""
	def __init__(self, global_vocabulary):
		self.population_array = []
		self.best_indivisual = None
		self.global_vocabulary = global_vocabulary

	def add(self, indivisual):
		self.population_array.append(indivisual)
		if self.best_indivisual == None or indivisual.getFitness() > self.best_indivisual.getFitness():
			self.best_indivisual = indivisual

	def size(self):
		return len(self.population_array)

	def getBestIndivisual(self):
		return self.best_indivisual

	def evolve(self):
		self.best_indivisual = None
		parents = self._parentSelection()
		for i in range(0, len(parents), 2):
			rnd = random.randint(0, 99)
			if rnd < CONSTANTS.CROSSOVER_RATE * 100:
				parents[i], parents[i+1] = self._crossover(parents[i], parents[i+1])
		for i in range(len(parents)):
			parents[i] = self._mutate(parents[i])
			if self.best_indivisual == None or parents[i].getFitness() > self.best_indivisual.getFitness():
				self.best_indivisual = parents[i]
		self.population_array = parents

	def _parentSelection(self):
		parents = []
		while len(parents) < len(self.population_array):
			tournament_population = random.sample(self.population_array, CONSTANTS.GENETIC_ALGORITHM_TOURNAMENT_SIZE)
			winner = max(tournament_population, key=lambda indivisual: indivisual.getFitness())
			parents.append(winner)

		return parents

	def _crossover(self, parent1, parent2):
		# 2 Point Crossover on vocabulary
		parent1_vocab = parent1.getVocabulary()
		parent2_vocab = parent2.getVocabulary()

		parent1_point1 = random.randint(0, len(parent1_vocab)-1)
		parent1_point2 = random.randint(0, len(parent1_vocab)-1)
		if parent1_point1 > parent1_point2:
			parent1_point1, parent1_point2 = parent1_point2, parent1_point1
		
		parent2_point1 = random.randint(0, len(parent2_vocab)-1)
		parent2_point2 = random.randint(0, len(parent2_vocab)-1)
		if parent2_point1 > parent2_point2:
			parent2_point1, parent2_point2 = parent2_point2, parent2_point1

		new_vocab1 = list(set(parent1_vocab[0:parent1_point1] + parent2_vocab[parent2_point1:parent2_point2+1] + parent1_vocab[parent1_point2+1:]))
		new_vocab2 = list(set(parent2_vocab[0:parent2_point1] + parent1_vocab[parent1_point1:parent1_point2+1] + parent2_vocab[parent2_point2+1:]))

		parent1.setVocabulary(new_vocab1)
		parent2.setVocabulary(new_vocab2)
		
		return parent1, parent2

	def _mutate(self, indivisual):
		# 1 Point Mutatiion on vocabulary
		indivisual_vocabulary = indivisual.getVocabulary()
		for word_index in range(len(indivisual_vocabulary)):
			rnd = random.randint(0, 99)
			if rnd < CONSTANTS.MUTATION_RATE * 100:
				indivisual_vocabulary[word_index] = random.choice(self.global_vocabulary)
		indivisual_vocabulary = list(set(indivisual_vocabulary))
		indivisual.setVocabulary(indivisual_vocabulary)
		return indivisual
