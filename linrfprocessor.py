'''
This script is dedicated to make Lin's
life easier so that every time she encounters
problem with random forest regression or classification,
she can finish it by simply anyixia!
'''

import numpy as np
import random
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from treeinterpreter import treeinterpreter as ti

class LinProcessor(object):
	"""You should Initialize this class with two
	numpy arrays and a list of feature names
	X: 2d-array [[],[]]
	y: 1d-array []
	names: []
	"""
	def __init__(self, X, y, names):
		self.X = X
		self.y = y
		self.names = names
		self.model = RandomForestRegressor().fit(self.X,self.y)

	def impurity(self):
		print ("Features sorted by their score under impurity criteria:")
		print (sorted(zip(map(lambda x: round(x, 4),
			self.model.feature_importances_), self.names),
			 reverse=True))

	def permuate(self):
		scores = defaultdict(list)
		for train_idx, test_idx in ShuffleSplit(len(self.X), 100, .3):
			X_train, X_test = self.X[train_idx], self.X[test_idx]
			Y_train, Y_test = self.y[train_idx], self.y[test_idx]
			r = self.model
			acc = r2_score(Y_test, self.model.predict(X_test))
			for i in range(self.X.shape[1]):
				X_t = X_test.copy()
			# permuate all the features
				np.random.shuffle(X_t[:, i])
				shuff_acc = r2_score(Y_test, self.model.predict(X_t))
				scores[self.names[i]].append((acc-shuff_acc)/acc)
		print ("Features sorted by their score under permuate criteria:")
		print (sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))
		return scores

	def treeinterpreter(self):
		# that [:300] can be changed as needed for now it's only setting for convenience
		# you can use [[200:300]] to find instances you're interested
		prediction, bias, contributions = ti.predict(self.model, self.X[:300])
		contributionsum = np.sum(contributions, axis=0)
		contributiondict = dict(zip(self.names,contributionsum))
		featurerank = sorted(contributiondict.items(), key=lambda kv: kv[1], reverse=True)
		print("Features sorted by their score under treeinterpreter criteria:")
		print(featurerank)
		return featurerank

def main():

	boston = load_boston()
	X = boston["data"]
	y = boston["target"]
	names = boston["feature_names"]

	Solution = LinProcessor(X,y,names)
	Solution.impurity()
	Solution.permuate()
	Solution.treeinterpreter()


if __name__ == "__main__":
	main()