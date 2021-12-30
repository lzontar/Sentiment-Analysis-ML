import random
import re
import string

import Levenshtein
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
import pandas as pd

import util.helper as helper
import os


def preprocess_data_minified(data_clean, target_col):

    data_clean = data_clean[data_clean['text'].notnull()]
    data_clean = data_clean[data_clean['text'].notna()]

    X = data_clean['text']
    y = data_clean[target_col]

    # We transform 4 to 1 to make our problem just a tad more beautiful :)
    y = pd.Series(list(map(lambda x: x if x == 0 else 1, y)))

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(X,
                                                            y,
                                                            train_size=0.8,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            shuffle=True,
                                                            stratify=None)

    X_train_.reset_index(drop=True, inplace=True)
    X_test_.reset_index(drop=True, inplace=True)
    y_train_.reset_index(drop=True, inplace=True)
    y_test_.reset_index(drop=True, inplace=True)

    return X_train_, X_test_, y_train_, y_test_


def preprocess_data(_data, target_col, preprocessing_dict, subsample_size, target_file):
    data_sample = _data.sample(frac=subsample_size/len(_data), replace=False, random_state=1)

    # We filter neutral data samples to simplify the problem into a binary classification problem
    data_sample = data_sample[data_sample[target_col] != 2]

    data_clean = data_sample.copy()
    data_clean['text'] = list(map(lambda x: preprocess_tweet(x, preprocessing_dict), data_clean['text']))

    # Remove retweets
    if helper.is_true_dict(preprocessing_dict, 'remove_retweets'):
        data_clean = remove_retweets(data_clean)

    data_clean.to_csv(target_file, encoding='latin-1')

    return data_clean


def preprocess_tweet(tweet, preprocessing_dict):
    tweet_clean = tweet

    # Basic preprocessing to try to make the text uniform
    # Remove non-ascii characters
    tweet_clean = ''.join(list(filter(lambda x: x in set(string.printable), tweet_clean)))

    # Remove URLs
    if helper.is_true_dict(preprocessing_dict, 'remove_urls'):
        tweet_clean = remove_urls(tweet_clean)

    # Remove hashtags
    if helper.is_true_dict(preprocessing_dict, 'remove_hashtags'):
        tweet_clean = remove_hashtags(tweet_clean)

    # Remove mentions
    if helper.is_true_dict(preprocessing_dict, 'remove_mentions'):
        tweet_clean = remove_mentions(tweet_clean)

    # Remove extra vowels
    if helper.is_true_dict(preprocessing_dict, 'remove_repetitive_vowels'):
        tweet_clean = remove_repetitive_vowels(tweet_clean)

    # Replace sequences of 'h' and 'a' with 'laugh'
    if helper.is_true_dict(preprocessing_dict, 'replace_laugh'):
        tweet_clean = replace_laugh(tweet_clean)

    # Replace emoticons to what they represent
    if helper.is_true_dict(preprocessing_dict, 'replace_emoji'):
        tweet_clean = replace_emoji(tweet_clean)

    # To lower case
    if helper.is_true_dict(preprocessing_dict, 'lower_case'):
        tweet_clean = str(tweet_clean).lower()

    # Remove punctuation
    tweet_clean = re.sub(r"(\.|,|:|;|\?|!|\)|\(|\-|\[|\]|\{|\}|\*|\||\<|\>|%|&|/|$|\+|@|#|\$|£|=|\^|~)", " ", tweet_clean)

    # Remove extra whitespaces
    tweet_clean = re.sub(r"\s+", " ", tweet_clean)

    # Filter stopwords
    if helper.is_true_dict(preprocessing_dict, 'filter_stopwords'):
        tweet_clean = filter_stopwords(tweet_clean)

    # Use dictionary to detect and correct misspellings (PyEnchant)
    if helper.is_true_dict(preprocessing_dict, 'fix_misspells'):
        tweet_clean = fix_misspells(tweet_clean)

    # Negations to 'not'
    if helper.is_true_dict(preprocessing_dict, 'handle_negations'):
        tweet_clean = handle_negations(tweet_clean)

    # Stemming
    if helper.is_true_dict(preprocessing_dict, 'stemming'):
        tweet_clean = stemming(tweet_clean)

    # Remove extra whitespaces
    tweet_clean = re.sub(r"\s+", " ", tweet_clean)

    return tweet_clean.strip()


def remove_urls(tweet):
    tweet_clean = tweet
    
    # Remove urls
    tweet_clean = re.sub(r"(http|https):\S+", " ", tweet_clean)

    return tweet_clean


def remove_hashtags(tweet):
    tweet_clean = tweet
    
    # Remove hashtags
    tweet_clean = re.sub(r"#\S+", " ", tweet_clean)
    return tweet_clean


def remove_mentions(tweet):
    tweet_clean = tweet
    
    # Remove mentions
    tweet_clean = re.sub(r"@\S+", " ", tweet_clean)
    return tweet_clean


def remove_repetitive_vowels(tweet):
    tweet_clean = tweet

    # Remove repetitions of more than 3 occurrences
    tweet_clean = re.sub("[a]{3,}", "aa", tweet_clean)
    tweet_clean = re.sub("[e]{3,}", "ee", tweet_clean)
    tweet_clean = re.sub("[i]{3,}", "ii", tweet_clean)
    tweet_clean = re.sub("[o]{3,}", "oo", tweet_clean)
    tweet_clean = re.sub("[u]{3,}", "uu", tweet_clean)

    return tweet_clean


def replace_laugh(tweet):
    tweet_clean = tweet

    # Transform the multiple occurrences of 'h' and 'a' to single occurrences
    tweet_clean = re.sub('hh+', "h", tweet_clean)
    tweet_clean = re.sub('aaa+', "a", tweet_clean)

    # Replace multiple occurrences of 'ha' and 'ah' to 'laugh'
    tweet_clean = re.sub(r"(ah){2,}|(ha){2,}", "laugh", tweet_clean)

    return tweet_clean


def replace_emoji(tweet):
    tweet_clean = tweet

    # Replace emoticons with emoticon combinations
    tweet_clean = tweet

    # angel, innocent
    tweet_clean = re.sub(r"O:\-\)|0:\-3| 0:3 |0:\-\)|0:\)|0;\^\)", " smile_angel ", tweet_clean)
    # evil
    tweet_clean = re.sub(r">:\)|>;\)|>:\-\)|\}:\-\)|}:\)|3:\-\)|3:\)", " smile_evil ", tweet_clean)
    # happy
    tweet_clean = re.sub(r":\-\)|:\)|:D|:o\)|:\]| :3|:c\)|:>| =\]|8\)|=\)|:\}|:\^\)", " smile_happy ", tweet_clean)
    # laugh
    tweet_clean = re.sub(r":\-D|8\-D|8D|x\-D|xD|X\-D|XD|=\-D|=D|=\-3| =3 |B\^D|:\-\)\)|:'\-\)|:'\)", " smile_laugh ", tweet_clean)
    # angry
    tweet_clean = re.sub(r":\-\|\||:@| >:\(", " smile_angry ", tweet_clean)
    # sad
    tweet_clean = re.sub(r">:\[|:\-\(|:\(|:\-c|:c|:\-<|:\-\[|:\[|:\{| <\\3 ", " smile_sad ", tweet_clean)
    # crying
    tweet_clean = re.sub(r";\(|:'\-\(|:'\(", " smile_crying ", tweet_clean)
    # horror, disgust
    tweet_clean = re.sub(r"D:<|D:|D8| D; |D=|DX|v\.v|D\-':", " smile_horror ", tweet_clean)
    # surprise, shock
    tweet_clean = re.sub(r">:O|:\-O|:O|:\-o|:o|8\-0|O_O|o\-o|O_o|o_O|o_o|O\-O", " smile_surprise ", tweet_clean)
    # kiss
    tweet_clean = re.sub(r":\*|:\-\*|:\^\*|\( '\}\{' \)|<3", " smile_kiss ", tweet_clean)
    # winking
    tweet_clean = re.sub(r";\-\)|;\)|\*\-\)|\*\)|;\-\]|;\]|;D|;\^\)|:\-,", " smile_wink ", tweet_clean)
    # tongue sticking out
    tweet_clean = re.sub(r">:P|:\-P|:P|X\-P|x\-p| xp | XP |:\-p| :p | =p |:\-Þ|:Þ|:þ|:\-þ|:\-b| :b | d: ", " smile_tongue_sticking_out ", tweet_clean)
    # skeptical
    tweet_clean = re.sub(r">:\\|>:/|:\-/|:\-\.|:/|:\\|=/|=\\| :L | =L | :S |>\.<", " smile_skeptical ", tweet_clean)
    # straight face
    tweet_clean = re.sub(r":\||:\-\|", " smile_straight_face ", tweet_clean)

    return tweet_clean


def lower_case(tweet):
    tweet_clean = tweet
    return tweet_clean


def stem_tweet(tweet):
    porter = PorterStemmer()

    token_words = word_tokenize(tweet)

    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def stemming(tweet):
    tweet_clean = tweet
    tweet_clean = stem_tweet(tweet)
    return tweet_clean


def filter_stopwords(tweet):
    tweet_clean = tweet

    stopwords_en = stopwords.words('english')
    tweet_clean = list(
        filter(
            lambda x: x not in stopwords_en,
            tweet_clean.split(' ')
        )
    )

    return ' '.join(tweet_clean)


def fix_misspells(tweet):
    spell = SpellChecker()
    tweet_clean = tweet

    # Find words that may be misspelled
    misspelled = spell.unknown(tweet_clean.split(' '))

    for word in misspelled:
        # Replace misspelled words with the most likely fix
        if word != '':
            tweet_clean = re.sub(word, spell.correction(word), tweet_clean)

    return tweet_clean


def replace_acronyms(tweet):
    tweet_clean = tweet

    with open('../data/preprocessing/acronyms.txt', 'r') as f_acronyms:
        for line in f_acronyms.readlines():
            split_line = line.split(' ')
            acronym = split_line[0].strip()
            replacement = split_line[1].strip()
            tweet_clean = re.sub(f" {acronym} ", f" {replacement} ", tweet_clean)

    return tweet_clean


def replace_cursewords(tweet):
    tweet_clean = tweet

    with open('../data/preprocessing/cursewords.txt', 'r') as f_cursewords:
        for line in f_cursewords.readlines():
            curseword = line.strip()
            tweet_clean = re.sub(f" {curseword} ", f" curseword ", tweet_clean)

    return tweet_clean


def handle_negations(tweet):
    tweet_clean = tweet

    tweet_clean = re.sub(r" isn't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" aren't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" don't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" didn't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" doesn't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" haven't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" hasn't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" wasn't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" weren't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" won't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" never ", " not ", tweet_clean)
    tweet_clean = re.sub(r" can't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" cannot ", " not ", tweet_clean)
    tweet_clean = re.sub(r" couldn't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" wouldn't ", " not ", tweet_clean)
    tweet_clean = re.sub(r" shouldn't ", " not ", tweet_clean)

    return tweet_clean


def remove_retweets(data):
    data_clean = data

    for ix, row_1 in data_clean.copy().iterrows():
        for iy, row_2 in data_clean.copy().iterrows():
            if ix != iy and Levenshtein.ratio(row_1['text'], row_2['text']) > 0.95:
                data_clean = data_clean[data_clean['tweet_id'] != row_2['tweet_id']]

    return data_clean
