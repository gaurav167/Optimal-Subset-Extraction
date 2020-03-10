import random

from nltk import word_tokenize
from nltk.util import ngrams

from data_loader import DataLoader
from preprocessor import Preprocessor
from indivisual import Indivisual
from population import Population
import CONSTANTS


def run(population, epochs):
	GENERATION = 1
	best_indivisual = population.getBestIndivisual()
	while GENERATION <= epochs:
		print("Running epoch number :", GENERATION)
		population.evolve()
		if population.getBestIndivisual().getFitness() > best_indivisual.getFitness():
			best_indivisual = population.getBestIndivisual()
		print("Generation", GENERATION, "Fitness :", population.getBestIndivisual().getFitness())
		GENERATION += 1
	print("Best Indivisual Fitness :", best_indivisual.getFitness())

train_data, test_data = DataLoader(CONSTANTS.SAMPLE_DATA_DIR).load()
# train_data, test_data = DataLoader(CONSTANTS.FULL_DATA_DIR).load()

print("Data loaded")

Preprocessor().preprocess(train_data, stemming=True)
Preprocessor().preprocess(test_data, stemming=True)

print("Preprocessing complete")

vocabulary = set()

for text in train_data['text']:
	tokens = word_tokenize(text)
	bigrams = ngrams(tokens, 2)
	for unigram in tokens:
		vocabulary.add(unigram)
	for bigram in bigrams:
		vocabulary.add(bigram)

vocabulary = list(vocabulary)
random.shuffle(vocabulary)
vocabulary_size = len(vocabulary)

indivisual_vocabulary_size = vocabulary_size//CONSTANTS.POPULATION_SIZE
assert(indivisual_vocabulary_size > 0)
start = 0

population = Population(vocabulary)
for indivisual_number in range(CONSTANTS.POPULATION_SIZE):
	end = vocabulary_size if indivisual_number + 1 == CONSTANTS.POPULATION_SIZE else start + indivisual_vocabulary_size
	indivisual_vocabulary = vocabulary[start:end]
	start = end
	indivisual = Indivisual(indivisual_vocabulary, train_data, test_data)
	population.add(indivisual)

print("Population initialized")

run(population, CONSTANTS.EPOCHS)