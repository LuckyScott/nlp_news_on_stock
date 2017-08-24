# -*- coding: utf-8 -*-

import yaml
import sys
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models.keyedvectors import  KeyedVectors

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.utils import np_utils

np.random.seed(1337)  # For Reproducibility
import jieba
import sys
import csv
import codecs
import os
import re
import pickle
import logging
import random
import json

sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 200
maxlen = 120
n_iterations = 1
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 120
cpu_count = multiprocessing.cpu_count()

# vectors_file = 'data/myvectors.bin'
vectors_file = 'data/vectors.bin'


def loadcsv(filename):
    data_list = []
    with codecs.open(filename, 'r', encoding='utf-8') as fd:
        reader = csv.reader(fd)
        for row in reader:
            data_list.append(row)
    print('load {} lines from file {}'.format(len(data_list), filename))
    return np.asarray(data_list)


def loadtext_file(filename):
    data_list = []
    with codecs.open(filename, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip(' \r\n\t\u3000')
            data_list.append(line.replace('\u3000', '').replace(' ', ''))
    print('load {} lines from file {}'.format(len(data_list), filename))
    return data_list


# split file separated by space (' ')
def load_split_file(filename):
    data_list = []
    with codecs.open(filename, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        for line in lines:
            data_list.append(line.split(' '))
        # reader = csv.reader(fd, delimiter=' ')  # fields too large, larger than limit 131072 ?
        # for row in reader:
        #     data_list.append(row)
    return data_list


def load_vectors(filename, binary: bool=False):
    model = KeyedVectors.load_word2vec_format(filename,binary=binary,unicode_errors='ignore')
    return model


#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [list(jieba.cut(document.replace('\n', ''))) for document in text]
    return text


def text_to_index_array(w2indx, sentences):  # 文本转为索引数字模式
    new_sentences = []
    for sen in sentences:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(w2indx[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)
    return sequence.pad_sequences(new_sentences, maxlen=maxlen)


#创建词语字典，并返回每个词语的索引，词向量
def create_dictionaries(model=None):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
    w2indx = {v: k+1 for k, v in gensim_dict.items()}
    w2vec = {word: model[word] for word in w2indx.keys()}
    return w2indx, w2vec


def save_word_index_dict(index_dict, word_vectors, filename):
    # 存储为pkl文件
    output = open(filename, 'wb')
    pickle.dump(index_dict, output)  # 索引字典
    pickle.dump(word_vectors, output)  # 词向量字典
    output.close()


def load_word_index_dict(filename):
    f = open(filename, 'rb')  # 预先训练好的
    index_dict = pickle.load(f)  # 索引字典，{单词: 索引数字}
    word_vectors = pickle.load(f)  # 词向量, {单词: 词向量(100维长的数组)}
    return index_dict, word_vectors


def get_data(index_dict,word_vectors,sentences,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2)
    print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=64, activation='sigmoid', inner_activation='hard_sigmoid'))
    # model.add(LSTM(output_dim=64, activation='relu',return_sequences=True)) # require 3 dimensions ?
    # model.add(LSTM(output_dim=64, activation='softplus'))
    model.add(Dropout(0.2))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # model.add(Activation('softmax'))
    # model.add(Activation('softplus'))
    model.add(Dense(5, activation='softmax'))
    model.summary()

    print('Compiling the Model...')
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Train...")
    # model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test),show_accuracy=True)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('data/lstm_w2v.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('data/lstm_w2v.h5')
    print('Test score:', score, 'Acc:', acc)


#训练模型，并保存
def train_model(w2vec_model, text, label):
    print('Tokenising...')
    sentences = tokenizer(text)
    print('Training a Word2vec model...')
    index_dict, word_vectors=create_dictionaries(w2vec_model)
    save_word_index_dict(index_dict, word_vectors, 'data/word2vec_index_dict.pkl')
    x_train = text_to_index_array(index_dict, sentences)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,x_train,label)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


def input_transform(sentences):
    index_dict, word_vectors = load_word_index_dict('data/word2vec_index_dict.pkl')
    x_text = text_to_index_array(index_dict, sentences)
    return x_text


def lstm_predict(sentences):
    print('loading model......')
    with open('data/lstm_w2v.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    print('loading weights......')
    model.load_weights('data/lstm_w2v.h5')
    # model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    data=input_transform(sentences)
    # print(len(data))
    result=model.predict_classes(data)
    return result


def train():
    train_file = 'data/train/res_content.txt'
    text = loadtext_file(train_file)
    print(len(text))
    # classify_file = 'data/train/res_price_classify.txt'
    classify_file = 'data/train/res_price_classify0.csv'
    y = loadcsv(classify_file)
    print(y.shape)
    y = y[:,6]
    print(y.shape)
    vector_model = load_vectors(vectors_file, binary=True)
    train_model(vector_model, text, y)


def predict():
    text_test = loadtext_file('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\Res_content.txt')
    out = codecs.open('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\Res_out.txt', 'w', encoding='utf-8')
    sentences = tokenizer(text_test)
    result = lstm_predict(sentences)
    logging.info('result is:' + '\n'.join([ str(x) for x in result]))
    print('result', result)
    for res in result:
        out.write(str(res) + '\n')
    out.close()


def predict2(test_file):
    sentences = load_split_file(test_file)
    result_file = os.path.dirname(test_file) + '/' + re.match('([a-zA-Z]+)', os.path.basename(test_file)).group() + '_predict_result.txt'
    out = codecs.open(result_file, 'w', encoding='utf-8')
    result = lstm_predict(sentences)
    logging.info('result is:' + '\n'.join([ str(x) for x in result]))
    print('result', result)
    for res in result:
        out.write(str(res) + '\n')
    out.close()
    return np.asarray(result).reshape(-1)


def get_classify_value(type):
    sign = 1 if random.random() >= 0.5 else -1
    if type == 0: # <= -0.03
        val = -0.03 - random.randint(0,30)/1000
        if random.random() >= 0.75:
            val -= random.random()/20
    elif type == 1: # -.03 ~ -.005
        val = -0.0175 + sign*(random.randint(0, 125) / 10000)
    elif type == 2: # -.005 ~ .005
        val = sign * (random.randint(0, 50) / 10000)
    elif type == 3: # .005 ~ .03
        val = 0.0175 + sign*(random.randint(0, 125) / 10000)
    else:
        val = 0.03 + random.randint(0,30)/1000
        if random.random() >= 0.75:
            val += random.random()/20
    return val


def convert_classify_to_value(uuid, data):
    result_list = []
    for i in range(len(uuid)):
        val1 = get_classify_value(data[i])
        val2 = get_classify_value(data[i])
        val3 = get_classify_value(data[i])
        res = {'uuid':uuid[i], 'value1':val1, 'value2':val2, 'value3':val3}
        result_list.append(res)
    return result_list


def save_predict_value(result_value, filename):
    with codecs.open(filename, 'w', encoding='utf-8') as fd:
        csv_writer = csv.DictWriter(fd, delimiter='\t', fieldnames=['uuid', 'value1', 'value2','value3'])
        csv_writer.writeheader()
        for row in result_value:
            csv_writer.writerow(row)


def new_predict(basedir, text_files, uuid_files, result_file):
    dir = basedir
    result_data = []
    for file in text_files:
        result = predict2(dir + file)
        result_data.extend(result)
    uuid_list = []
    for file in uuid_files:
        with codecs.open(dir + file, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
            for line in lines:
                uuid_list.append(line.strip(' \r\n'))
    result_values = convert_classify_to_value(uuid_list, result_data)
    # logging.info('res:' + json.dumps(result_values))
    save_predict_value(result_values, dir + result_file)


def predict_T1():
    text_files = ['res_title_content_split.txt', 'ann_title_content_split.txt']
    uuid_files = ['Res_uuid.txt', 'Ann_uuid.txt']
    basedir = 'data/test/T1/'
    result_file = 'predict_result.txt'
    new_predict(basedir, text_files, uuid_files, result_file)


def predict_T2():
    text_files = ['res_title_content_split.txt', 'ann_title_content_split.txt', 'New_title_content_split.txt']
    uuid_files = ['Res_uuid.txt', 'Ann_uuid.txt', 'New_uuid.txt']
    basedir = 'data/test/T2/'
    result_file = 'predict_result.txt'
    new_predict(basedir, text_files, uuid_files, result_file)


# res + ann , title+content
def train3():
    train_file = 'data/train/res_title_content_split.txt'
    logging.info('loading train file: {}'.format(train_file))
    res_text = load_split_file(train_file)
    logging.info('load {} lines from file: {}'.format(len(res_text), train_file))

    train_file = 'data/train/ann_title_content_split-1.txt'
    logging.info('loading train file: {}'.format(train_file))
    ann_text = load_split_file(train_file)
    logging.info('load {} lines from file: {}'.format(len(ann_text), train_file))

    sentences = res_text + ann_text
    # sentences = ann_text
    logging.info('merge research an announce, total {} lines'.format(len(sentences)))

    classify_file = 'data/train/res_price_classify.csv'
    logging.info('loading classify file: {}'.format(classify_file))
    res_y = loadcsv(classify_file)
    logging.info('classify {} shape is: {}'.format(classify_file, res_y.shape))

    classify_file = 'data/train/ann_price_classify-1.csv'
    logging.info('loading classify file: {}'.format(classify_file))
    ann_y = loadcsv(classify_file)
    logging.info('classify {} shape is: {}'.format(classify_file, ann_y.shape))

    y = np.append(res_y,ann_y, 0)
    print(y.shape)
    y = y[:, 12]
    print(y.shape)
    y = np_utils.to_categorical(y)
    # vector_model = load_vectors(vectors_file, binary=True)
    # print('Training a Word2vec model...')
    # index_dict, word_vectors=create_dictionaries(w2vec_model)
    # save_word_index_dict(index_dict, word_vectors, 'data/word2vec_index_dict.pkl')
    index_dict, word_vectors = load_word_index_dict('data/word2vec_index_dict.pkl')
    x_text = text_to_index_array(index_dict, sentences)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,x_text,y)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


def main():
    logging.basicConfig(filename='logger.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #################################################################################################
    # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s : %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################

    # train()
    # train3()
    # predict()

    predict_T1()
    predict_T2()


if __name__=='__main__':
    main()
