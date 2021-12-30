import random
import time

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from util.models.IClassifier import IClassifier
from util import helper


class NaiveBayes_(IClassifier):
    def __init__(self, n_splits=10):
        self.label = 'NaiveBayes'
        self.model = MultinomialNB()
        self.random_state = 1
        self.n_splits = n_splits
        self.vectorizer = CountVectorizer(stop_words='english')

    def prepDataClassifier(self, X_train, y_train, X_test, y_test):
        whole_data = pd.concat([X_train, X_test])
        self.vectorizer.fit(whole_data)

        X_train_, y_train_ = self.prepDataClassifierSingle(X_train, y_train)
        X_test_, y_test_ = self.prepDataClassifierSingle(X_test, y_test)

        return X_train_, y_train_, X_test_, y_test_

    def prepDataClassifierSingle(self, X, y):
        X_ = self.vectorizer.transform(X).toarray()
        y_ = y
        return X_, y_

    def fitClassifierInner(self, X_train, y_train, X_test, y_test):
        X_train, y_train, X_test, y_test = self.prepDataClassifier(X_train, y_train, X_test, y_test)

        self.model.fit(X_train, y_train)

    def predict(self, X, y):
        X_, y_ = self.prepDataClassifierSingle(X, y)
        return list(map(lambda x: int(x), self.model.predict(X_)))
