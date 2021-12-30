from util import helper
from collections import Counter


class MajorityClassifier_:
    def __init__(self, n_splits=10):
        self.train = []
        self.test = []

        self.cv = {}
        for i in range(n_splits):
            self.cv[str(i)] = {
                'train': [],
                'test': []
            }

    def addToTrain(self, results):
        self.train.append(results)

    def addToTest(self, results):
        self.test.append(results)

    def addToCv(self, ix, results, dataset):
        self.cv[str(ix)][dataset].append(results)

    def getResults(self, y_train, y_test, y_cv_list):
        y_pred_train = []
        for ix in range(len(self.train[0])):
            vote = Counter(list(map(lambda x: x[ix], self.train))).most_common(1)[0][0]
            y_pred_train.append(vote)
        y_pred_test = []
        for ix in range(len(self.test[0])):
            vote = Counter(list(map(lambda x: x[ix], self.test))).most_common()[0][0]
            y_pred_test.append(vote)

        results = {
            'train': helper.calcScores(y_train, y_pred_train),
            'test': helper.calcScores(y_test, y_pred_test)
        }
        for dataset in ['train', 'test']:
            results[f'cv_{dataset}'] = []
            for ix, y_cv_i in enumerate(y_cv_list[dataset]):
                y_pred_cv_i = []
                for iy in range(len(self.cv[str(ix)][dataset][0])):
                    vote = Counter(list(map(lambda x: x[iy], self.cv[str(ix)][dataset]))).most_common()[0][0]
                    y_pred_cv_i.append(vote)
                results[f'cv_{dataset}'].append(helper.calcScores(y_pred_cv_i, y_cv_i))

        return results
