import gc
import os
import pickle
import time
from itertools import tee

import en_core_web_sm
import pandas as pd
import tensorflow as tf
import torch
from numba import cuda
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

sns.set_theme()
sns.color_palette("Set2")
import util.helper as helper
import util.helper_preprocessing as preprocessing
from util.models.BERT import BERT_
from util.models.LSTM import LSTM_
from util.models.MajorityClassifier_ import MajorityClassifier_
from util.models.NaiveBayes import NaiveBayes_

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def getModelInstance(name, word_to_vec_map_, primary=False):
    if name == 'NaiveBayes': return NaiveBayes_()
    if name == 'BERT': return BERT_(primary=primary)
    if name == 'LSTM': return LSTM_(word_to_vec_map=word_to_vec_map_, plot=primary)


if __name__ == '__main__':
    take_preprocessed = True
    generate_figures = True
    generate_models = False

    show_results = False

    model_names = ['NaiveBayes', 'BERT', 'LSTM']
    subsample_size = 40000
    word_to_vec_map = helper.read_glove_vector('../data/glove/glove.twitter.27B.200d.txt')

    target_file = f'../data/preprocessing/final_data_{subsample_size}.csv'
    if os.path.isfile(target_file) and take_preprocessed:
        print(f"Taking preprocessed dataset of {subsample_size} samples")
        data_clean = pd.read_csv(target_file, encoding='latin-1')
    else:
        start_time = time.time()
        data = helper.read_data(data_path='../data/training.1600000.processed.noemoticon.csv',
                                col_list=['target', 'tweet_id', 'date', 'flag', 'user', 'text'])
        read_time = time.time()
        print("(timer) readData executed in: %s seconds" % (read_time - start_time))

        preprocess_dict = {
            'remove_urls': True,
            'remove_hashtags': True,
            'remove_mentions': True,
            'remove_repetitive_vowels': True,
            'replace_laugh': True,
            'replace_emoji': True,
            'lower_case': True,
            'stemming': False,
            'filter_stopwords': True,
            'fix_misspells': False,
            'handle_negations': False,
            'remove_retweets': False
        }

        data_clean = preprocessing.preprocess_data(data, 'target', preprocess_dict,
                                                   subsample_size=subsample_size,
                                                   target_file=target_file)

        preprocess_time = time.time()
        print("(timer) preprocessingData: %s seconds" % (preprocess_time - read_time))

    X_train, X_test, y_train, y_test = preprocessing.preprocess_data_minified(data_clean, 'target')

    if generate_figures:
        helper.visualizeData(X_train, y_train)

    if generate_models:
        n_splits = 10
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        splits = cv.split(X_train, y_train)
        results = {}

        majority_classifier = MajorityClassifier_(n_splits=n_splits)
        y_cv_list = {'train': [], 'test': []}
        for model_name in model_names:
            model_results = {}
            model = getModelInstance(model_name, word_to_vec_map, primary=True)

            model.fitClassifier(X_train, y_train, X_test, y_test)

            y_pred_train, res_train = helper.scorer(model, X_train, y_train)
            majority_classifier.addToTrain(y_pred_train)
            y_pred_test, res_test = helper.scorer(model, X_test, y_test)
            majority_classifier.addToTest(y_pred_test)
            model_results['train'] = res_train
            model_results['test'] = res_test
            model_results['cv_train'] = []
            model_results['cv_test'] = []

            # Reset GPU memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

            res_cv = {}
            splits, splits_copy = tee(splits)
            for ix, (train_index, test_index) in enumerate(splits_copy):
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

                if len(y_cv_list['train']) < n_splits:
                    y_cv_list['train'].append(y_train_cv)

                if len(y_cv_list['test']) < n_splits:
                    y_cv_list['test'].append(y_test_cv)

                # Reset GPU memory
                if ix > 0:
                    del cv_model
                gc.collect()
                torch.cuda.empty_cache()

                cv_model = getModelInstance(model_name, word_to_vec_map)
                cv_model.fitClassifierInner(X_train_cv, y_train_cv, X_test_cv, y_test_cv)

                y_cv_train_pred, res_train_cv = helper.scorer(cv_model, X_train_cv, y_train_cv)
                y_cv_test_pred, res_test_cv = helper.scorer(cv_model, X_test_cv, y_test_cv)
                majority_classifier.addToCv(ix, y_cv_train_pred, 'train')
                model_results['cv_train'].append(res_train_cv)
                majority_classifier.addToCv(ix, y_cv_test_pred, 'test')
                model_results['cv_test'].append(res_test_cv)

            with open(f'../data/results/{model_name}_{subsample_size}.pkl', 'wb+') as f:
                pickle.dump(model_results, f)
                print(model_results)
        with open(f'../data/results/MajorityClassifier_{subsample_size}.pkl', 'wb+') as f:
            pickle.dump(majority_classifier.getResults(y_train, y_test, y_cv_list), f)
            print(majority_classifier.getResults(y_train, y_test, y_cv_list))
    if show_results:
        helper.showResults(model_names, subsample_size)
