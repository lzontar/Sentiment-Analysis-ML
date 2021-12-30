import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wordcloud
from sklearn import metrics
import random
import time
import seaborn as sns


def is_true_dict(dict, key):
    return key in dict.keys() and dict[key]


def read_data(data_path, col_list):
    _data = pd.read_csv(data_path, encoding='latin-1')
    _data.columns = col_list
    return _data


def scorer(estimator, X, y):
    y_pred = estimator.predict(X, y)
    return y_pred, calcScores(y, y_pred)


def calcScores(y, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y, y_pred),
        'precision': metrics.precision_score(y, y_pred),
        'recall': metrics.recall_score(y, y_pred),
        'f1_score': metrics.f1_score(y, y_pred, zero_division=1)
    }



def showResults(alg_labels, sample_size):
    name_mapper = {
        'LSTM': 'LSTM',
        'MajorityClassifier': 'Majority classifier',
        'BERT': 'BERT',
        'NaiveBayes': 'Naive Bayes'
    }

    metric_mapper = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 score'
    }

    alg_labels.append('MajorityClassifier')

    results_test = []
    results_table = []
    for label in alg_labels:
        with open(f'../data/results/{label}_{sample_size}.pkl', 'rb') as f:
            model_results = pickle.load(f)
        cell_table = {'Algorithm': name_mapper[label]}
        for metric in metric_mapper.keys():

            metric_list = list(map(lambda x: x[metric], model_results['cv_test']))
            metric_list_train = list(map(lambda x: x[metric], model_results['cv_train']))

            cell_table[f'{metric}_train'] = model_results['train'][metric]
            cell_table[f'{metric}_test'] = model_results['test'][metric]
            cell_table[f'{metric}_cv_train_mean'] = np.mean(metric_list_train)
            cell_table[f'{metric}_cv_train_sd'] = np.std(metric_list_train)
            cell_table[f'{metric}_cv_test_mean'] = np.mean(metric_list)
            cell_table[f'{metric}_cv_test_sd'] = np.std(metric_list)

            for ix_val in range(len(metric_list)):
                val = metric_list[ix_val]
                val_relative = ((metric_list_train[ix_val] - metric_list[ix_val]) / metric_list_train[ix_val]) * 100
                df_item = {'Algorithm': name_mapper[label], 'Evaluation metric': metric_mapper[metric], 'Score': val, 'Decrease on new data (%)': val_relative}

                results_test.append(df_item)
        results_table.append(cell_table)

    df_res = pd.DataFrame(results_test)
    sns.barplot(x="Algorithm", y="Score", hue="Evaluation metric", data=df_res)
    plt.ylim([0.7, 0.8])
    plt.title('Results - 10-fold CV validation set')
    plt.savefig('../data/results/fig/cv_test.png')
    plt.show()
    plt.close()

    sns.barplot(x="Algorithm", y="Decrease on new data (%)", hue="Evaluation metric", data=df_res)
    plt.title('Results - 10-fold CV - overfitting')
    plt.savefig('../data/results/fig/cv_overfitting.png')
    plt.show()
    plt.close()

    print(results_table)

def toXy(data):
    X = data.loc[:, data.columns != 'target']
    y = data['target']
    return X, y


def fromXy(X, y):
    data = pd.concat([pd.DataFrame(X, index=y.index), pd.Series(y)], axis=1).reset_index(drop=True)
    data.columns = list(map(lambda x: f'text_{x}', range(X.shape[1]))) + ['target']
    return data


def visualizeData(X, y):
    sns.color_palette("Set2")

    # Wordcloud
    common_words = ''
    for index, value in X.items():
        tokens = str(value).split()
        common_words += " ".join(tokens) + " "
    wordcloud_ = wordcloud.WordCloud().generate(common_words)
    plt.imshow(wordcloud_, interpolation='bilinear')
    plt.axis("off")
    plt.title('Wordcloud - most common words')
    plt.savefig('../data/results/fig/wordcloud.png')
    plt.show()

    # Preprocessed tweet length distribution
    lengths = list(map(lambda x: len(x.split()), list(X)))
    df = pd.DataFrame()
    df['Preprocessed tweet length'] = lengths
    sns.displot(df, x="Preprocessed tweet length", bins=24, kde=True)
    plt.title('Preprocessed tweet length distribution')
    plt.tight_layout()
    plt.savefig('../data/results/fig/tweet_len_distr.png')
    plt.show()
    plt.close()

    # Sentiment  distribution
    neg = list(filter(lambda x: x == 0, list(y)))
    df = pd.DataFrame({
        'Text polarity': ['Positive', 'Negative'],
        'Count': [len(y) - len(neg), len(neg)]
    })
    sns.barplot(x=df['Text polarity'], y=df['Count'])
    plt.title('Text polarity distribution')
    plt.savefig('../data/results/fig/text_polarity_distr.png')
    plt.show()
    plt.close()


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

    return word_to_vec_map
