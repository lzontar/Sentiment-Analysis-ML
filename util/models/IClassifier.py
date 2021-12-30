import time
from abc import abstractmethod


class IClassifier:
    @abstractmethod
    def prepDataClassifier(self, X_train, y_train, X_test, y_test):
        pass

    def fitClassifier(self, X_train, y_train, X_test, y_test):
        start_time = time.time()
        self.fitClassifierInner(X_train, y_train, X_test, y_test)
        end_time = time.time()
        print("(timer) fitClassifier executed in: %s seconds" % (end_time - start_time))

    @abstractmethod
    def fitClassifierInner(self, X_train, y_train, X_test, y_test):
        pass

    def evaluateClassifier(self, X_train, y_train, X_test, y_test):
        start_time = time.time()
        results = self.evaluateClassifierInner(X_train, y_train, X_test, y_test)
        end_time = time.time()
        print("(timer) evaluateClassifier executed in: %s seconds" % (end_time - start_time))
        return results

    @abstractmethod
    def evaluateClassifierInner(self, X_train, y_train, X_test, y_test):
        pass

