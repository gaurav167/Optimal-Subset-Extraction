from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class Model:
	"""Machine Learning/NLP model for Ngram evaluation"""
	"""SVC Model in this case"""
	def __init__(self, vocabulary, train_data, test_data):
		self.vocabulary = vocabulary
		self.train_data = train_data
		self.test_data = test_data
		vectorizer = TfidfVectorizer(vocabulary=self.vocabulary)
		training_features = vectorizer.fit_transform(self.train_data['text'])
		self.test_features = vectorizer.transform(self.test_data['text'])
		self.model = LinearSVC()
		self.model.fit(training_features, self.train_data['sentiment'])

	def calcAccuracy(self):
		self.y_pred = self.model.predict(self.test_features)
		accuracy = accuracy_score(self.test_data['sentiment'], self.y_pred)
		return accuracy