#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
import logging
import tkinter.filedialog
import jieba
import codecs
import os

import numpy as np
np.random.seed(1337)  # For Reproducibility

from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.corpora.dictionary import Dictionary

vocab_dim = 100
maxlen = 400
window_size = 7
batch_size = 32


# 创建词语字典，并返回word2vec模型中词语的索引，词向量
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: p_model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec


def loadfile(filename):
    data_list = []
    with codecs.open(filename, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip(' \r\n\t\u3000')
            data_list.append(line.replace('\u3000', '').replace(' ', ''))
    print('load {} lines from file {}'.format(len(data_list), filename))
    return data_list


def save_split_text(data_list, filename):
    with codecs.open(filename, 'w', encoding='utf-8') as fd:
        for line in data_list:
            fd.writelines(' '.join(line) + '\n')
    print('saved {} lines to file {}'.format(len(data_list), filename))


def split_words(sentences):
    # new_sentences = [list(jieba.cut(line)) for line in sentences]
    new_sentences = []
    line_num = 0
    total = len(sentences)
    for line in sentences:
        new_sentences.append(list(jieba.cut(line)))
        line_num += 1
        if line_num % 5000 == 0:
            print('split {}% of {}'.format(line_num/total*100, total))
            logging.info('split {}% of {}'.format(line_num/total*100, total))

    return new_sentences


# 主程序
logging.basicConfig(filename='logger.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = []
# load research title
logging.info('load file res_title.txt')
text = loadfile(os.path.dirname(__file__) + '/data/train/res_title.txt')
text = split_words(text)
save_split_text(text, os.path.dirname(__file__) + '/data/train/res_title_split.txt')
sentences.extend(text)

# load research content
logging.info('load file res_content.txt')
text = loadfile(os.path.dirname(__file__) + '/data/train/res_content.txt')
text = split_words(text)
save_split_text(text, os.path.dirname(__file__) + '/data/train/res_content_split.txt')
sentences.extend(text)

# load announce title
logging.info('load file ann_annonce_title.txt')
text = loadfile(os.path.dirname(__file__) + '/data/train/ann_annonce_title.txt')
text = split_words(text)
save_split_text(text, os.path.dirname(__file__) + '/data/train/ann_title_split.txt')
sentences.extend(text)

# # load announce content
# text = loadfile(os.path.dirname(__file__) + '/data/train/ann_content.txt')
# text = split_words(text)
# save_split_text(text, os.path.dirname(__file__) + '/data/train/ann_content_split.txt')
# sentences.extend(text)


logging.info('训练Word2vec模型')
print('训练Word2vec模型（可尝试修改参数）...')
model = Word2Vec(sentences,
                 size=vocab_dim,  # 词向量维度
                 min_count=5,  # 词频阈值
                 window=window_size)  # 窗口大小
print('save model....')
logging.info('save Word2vec模型')
model.save('word2vec.model')

print('save vectors...')
model_wv = model.wv
model_wv.save_word2vec_format('myvectors.bin',binary=True)
model_wv.save_word2vec_format('myvectors.txt',binary=False)

print('vocab vector_size:{}  vocab size:{}'.format(model_wv.vector_size, len(model_wv.vocab)))

# 索引字典、词向量字典
index_dict, word_vectors= create_dictionaries(model_wv)

print(' word_verctors size:{} {}'.format(len(word_vectors), len(word_vectors['是'])))

# 存储为pkl文件
# pkl_name = raw_input(u"请输入保存的pkl文件名...\n").decode("utf-8")
output = open("word2vec.model.pkl", 'wb')
pickle.dump(index_dict, output)  # 索引字典
pickle.dump(word_vectors, output)  # 词向量字典
output.close()

if __name__ == "__main__":
    pass
