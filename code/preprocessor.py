import re
import unidecode

import pandas as pd
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Preprocessor():
	"""Utility text preprocessing class"""
	
	notalnum = re.compile('[\W_]+', re.UNICODE)
	stop_words = set(stopwords.words('english'))
	stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '', '\n'])
	porter_stemmer = PorterStemmer()

	def preprocess(self, df, stemming=False):
		df['text'] = df['text'].apply(Preprocessor._clean)
		if stemming:
			df['text'] = df['text'].apply(Preprocessor._stem)

	@staticmethod
	def _clean(text):
		words = sent_tokenize(text)
		clean_words = []
		for word in words:
			word = wordpunct_tokenize(word.lower())
			word = [unidecode.unidecode(Preprocessor.notalnum.sub('' , word)) for word in word if not word.isdigit() and word not in Preprocessor.stop_words]
			clean_words += list(filter(None, word))
		clean_words = ' '.join(clean_words)
		return clean_words
	
	@staticmethod
	def _stem(text):
		words = wordpunct_tokenize(text.lower())
		words = [Preprocessor.porter_stemmer.stem(word) for word in words]
		words = ' '.join(words)
		return words
