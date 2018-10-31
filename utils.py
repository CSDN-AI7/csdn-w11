#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    ##################
    # 将字符型vocabulary列表编码为整数vocabulary_int列表
    vocabulary_int = vocabulary_to_inter(vocabulary)

    #  每次从data中取出（batch_size行，num_steps列）个元素，每个元素都是整数
    t = ( len(vocabulary_int) // (batch_size*num_steps) )* (batch_size*num_steps)   # 防止data长度不能被batch_size、num_steps整除
    data_x = vocabulary_int[0:t]                 # 取出对应x长度数据
    data_y = np.zeros_like(data_x)
    data_y[ :-1], data_y[-1] = data_x[1:], data_x[0]   # Y数据是x数据右移一位的结果

    x_batch = data_x.reshape((batch_size, -1))      # reshape成batch_size行，每一行再拆分成若干个num_steps
    y_batch = data_y.reshape((batch_size, -1))      # reshape成batch_size行，每一行再拆分成若干个num_steps

    while True:
        for n in range(0, x_batch.shape[1], num_steps):  # 以num_steps步进
            x = x_batch[:, n:n + num_steps]           # 行表示batch_size，列表示num_steps
            y = y_batch[:, n:n + num_steps]           # 行表示batch_size，列表示num_steps
            yield x, y  

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
