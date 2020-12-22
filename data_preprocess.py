from utils import load_data, save_pickle, load_pickle, split_to_dict

import numpy as np
import pandas as pd

import re
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
import itertools

from keras.preprocessing.sequence import pad_sequences

# convert sentence to list of words
def sentence_to_word_list(sentence):
    sentence = str(sentence).lower()
    word_list = sentence.split()

    return word_list

# build embedding matrix
def build_embedding(train_data, test_data, dim=300):
    vocab = {}
    inv_vocab = ['<unk>']
    
    embedding = './embedding/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(embedding, binary=True)

    stops = set(stopwords.words('english'))

    sentence_cols = ['sentence_A', 'sentence_B']

    for data in [train_data, test_data]:
        for i, row in data.iterrows():
            for sentence in sentence_cols:
                num_rep = []
                for word in sentence_to_word_list(row[sentence]):
                    # filter out stop words
                    if word in stops and word not in word2vec.vocab:
                        continue
                    
                    if word not in vocab:
                        num = len(inv_vocab)
                        vocab[word] = num
                        num_rep.append(num)
                        inv_vocab.append(word)
                    else:
                        num_rep.append(vocab[word])
                data.at[i, sentence] = num_rep
    
    embedding_matrix = 1 * np.random.randn(len(vocab) + 1, dim)
    embedding_matrix[0] = 0

    for word, idx in vocab.items():
        if word in word2vec.vocab:
            embedding_matrix[idx] = word2vec.word_vec(word)
    
    del word2vec
    return embedding_matrix, train_data, test_data

# process data into left and right, normalize values, split into train/val/test
def prepare_data(train_data, test_data):
    max_seq_length = max(train_data['sentence_A'].map(lambda x: len(x)).max(),
                     train_data['sentence_B'].map(lambda x: len(x)).max(),
                     test_data['sentence_A'].map(lambda x: len(x)).max(),
                     test_data['sentence_B'].map(lambda x: len(x)).max(),)

    sentence_cols = ['sentence_A', 'sentence_B']
    X = train_data[sentence_cols]
    y = train_data['relatedness_score']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    X_train = split_to_dict(X_train, sentence_cols)
    X_val = split_to_dict(X_val, sentence_cols)
    X_test = split_to_dict(test_data, sentence_cols)
    y_test = test_data['relatedness_score']

    # normalize scores between 0-1 range
    y_train /= 5
    y_val /= 5
    y_test /= 5

    for data, split in itertools.product([X_train, X_val, X_test], ['left', 'right']):
        data[split] = pad_sequences(data[split], maxlen=max_seq_length)

    train = [X_train, y_train]
    val = [X_val, y_val]
    test = [X_test, y_test]

    return train, val, test, max_seq_length

if __name__ == '__main__':
    train_data_dir = '../../data/SICK_train.txt'
    test_data_dir = '../../data/SICK_test.txt'

    train_data = load_data(train_data_dir)
    test_data = load_data(test_data_dir)

    embedding_matrix, train_data, test_data = build_embedding(train_data, test_data)

    save_pickle(embedding_matrix, './embedding/SICK_matrix')