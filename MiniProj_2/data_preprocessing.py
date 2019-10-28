# df len = 70000
# consider this scirpt. takes test features as well in the input


import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# init stemmer
porter_stemmer = PorterStemmer()


def data_preparation():
    pass


def lemmatizer(sentence):
    x = [porter_stemmer.stem(word=word) for word in str(sentence).split(' ')]
    y = ""
    for i in x:
        y = y + i + " "
    return y.strip()


DELETE_CHARS = '!@#$%^*()+=?\'\"{}[]<>!`:;|\\/-,.&'


def fix_string(s, old='', new=''):
    table = s.maketrans(old, new, DELETE_CHARS)
    return s.translate(table)


def remove_numbers(text):
    output = ''.join(c for c in text if not c.isdigit())
    return output


def stop_wrodremoval(s, stop):
    words = s.split(" ")
    shortlisted_words = []

    # remove stop words
    for w in words:
        if w not in stop:
            shortlisted_words.append(w.strip())
    return ' '.join(shortlisted_words)


wordnet_lemmatizer = WordNetLemmatizer()
import nltk


def nltk_lemmatizer(text):
    stopword = stopwords.words("english")
    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    return " ".join(lemmatized_word)


def word_tokenize(text):
    word_tokens = nltk.word_tokenize(text)
    return " ".join(word_tokens)


import pandas


def main(df, df1):
    # df = pandas.read_csv("/Users/Srikanth/PycharmProjects/MiniProject1/miniproj2/dataset/reddit_train.csv")
    # df1 = pandas.read_csv("/Users/Srikanth/PycharmProjects/MiniProject1/miniproj2/dataset/reddit_test.csv")
    df1['subreddits'] = None
    df = df.append(df1, ignore_index=True)
    data_mod = df
    stop = stopwords.words('english')

    # Quantization

    data_mod = data_mod[pd.notnull(data_mod['comments'])]
    data_mod['comments'] = data_mod['comments'].str.lower()
    # remove unnecessary symbols, links, new line chars
    data_mod['comments'] = data_mod['comments'].apply(lambda x: re.sub('_', ' ', x))
    data_mod['comments'] = data_mod['comments'].apply(lambda x: re.sub(r'@\w+', '', x))
    data_mod['comments'] = data_mod['comments'].apply(lambda x: re.sub(r'http.?://[^\s]+[\s]_?', '', x))
    data_mod['comments'] = data_mod['comments'].apply(lambda x: re.sub(r'\n', ' ', x))
    data_mod['comments'] = data_mod['comments'].apply(lambda x: remove_numbers(x))

    # removing the Punctuation
    # remove stop words

    # removing the stemming words ex: singing -> sing
    data_mod['comments'] = data_mod['comments'].apply(lambda x: stop_wrodremoval(x, stop))
    data_mod['comments'] = data_mod['comments'].apply(lambda x: word_tokenize(x))
    data_mod['comments'] = data_mod['comments'].apply(lambda x: nltk_lemmatizer(x))
    # data_mod['comments'] = data_mod['comments'].apply(lambda x: lemmatizer(x))
    data_mod['comments'] = data_mod['comments'].apply(lambda x: fix_string(x))
    return data_mod.head(70000), data_mod.tail(30000)
