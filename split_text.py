#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
import logging
import jieba
import codecs
import re
import os
import numpy as np


def convert_utf8bom(filename):
    """
    处理文件中有u'\ufeff'的问题(utf-8 BOM)
    """
    fd = codecs.open(filename,'r', encoding='utf-8')
    if fd.read(1) == '\ufeff':
        # is utf-8 bom
        s = fd.read()
        fd.close()
        codecs.open(filename, mode='w', encoding='utf-8').write(s)
        print(filename+'   UTF8无BOM格式转换完成!!')
    else:
        fd.close()
        print(filename+' 不是 UTF8 +BOM格式, 无需转换!!')


def loadfile(filename):
    data_list = []
    with codecs.open(filename, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip(' \r\n\t\u3000')
            line = re.sub('[a-zA-Z\d\s\]\[＝×－÷…○●〔〕＆§∨「」＿+≥／≤‖※◆．～–％√□〈〉\u3000·~\!@#$%^&*/<>(),.;:，。“”‘’\'\"：；∞\{\}〃℃︵︶★『』￥━㎡《》】【！（）?？、_\-=—]*','', line)
            # data_list.append(line.replace('\u3000', '').replace(' ', '').replace('\t', ''))
            data_list.append(line)
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


def split_words_and_save(sentences, save_filename):
    line_num = 0
    total = len(sentences)
    with codecs.open(save_filename, 'w', encoding='utf-8') as fd:
        for line in sentences:
            words = list(jieba.cut(line))
            fd.write(' '.join(words) + '\n')
            line_num += 1
            if line_num % 5000 == 0:
                print('split {}% of {}'.format(line_num*100//total, total))
                logging.info('split {}% of {}'.format(line_num*100//total, total))
        print('split finished!')


def split_test_files(t_type='T1'):
    # load research title
    logging.info('load file Res_title.txt')
    text_title = loadfile('data/test/'+ t_type + '/Res_title.txt')
    # load research content
    logging.info('load file Res_content.txt')
    text_content = loadfile('data/test/' + t_type + '/Res_content.txt')
    sentences = []
    for i in range(len(text_title)):
        if text_content[i] == 'NULL':
            sentences.append(text_title[i])
        else:
            sentences.append(text_title[i] + text_content[i])
    split_words_and_save(sentences, 'data/test/' + t_type + '/res_title_content_split.txt')

    logging.info('load file Ann_annonce_title.txt')
    text_title = loadfile('data/test/' + t_type +'/Ann_annonce_title.txt')
    # load research content
    logging.info('load file Ann_content.txt')
    text_content = loadfile('data/test/' + t_type + '/Ann_content.txt')
    sentences = []
    for i in range(len(text_title)):
        if text_content[i] == 'NULL':
            sentences.append(text_title[i])
        else:
            sentences.append(text_title[i] + text_content[i])
    split_words_and_save(sentences, 'data/test/' + t_type + '/ann_title_content_split.txt')

    if t_type != 'T1':
        logging.info('load file New_title.txt')
        text_title = loadfile('data/test/' + t_type + '/New_title.txt')
        # load research content
        logging.info('load file New_content.txt')
        text_content = loadfile('data/test/' + t_type + '/New_content.txt')
        sentences = []
        for i in range(len(text_title)):
            if text_content[i] == 'NULL':
                sentences.append(text_title[i])
            else:
                sentences.append(text_title[i] + text_content[i])
        split_words_and_save(sentences, 'data/test/' + t_type + '/New_title_content_split.txt')


def main():
    # load research title
    logging.info('load file res_title.txt')
    text_title = loadfile('data/train/res_title.txt')
    logging.info('load file res_content.txt')
    text_content = loadfile('data/train/res_content.txt')
    sentences = []
    for i in range(len(text_title)):
        if text_content[i] == 'NULL':
            sentences.append(text_title[i])
        else:
            sentences.append(text_title[i] + text_content[i])
    split_words_and_save(sentences, 'data/train/res_title_content_split.txt')

    # load announce
    logging.info('load file ann_annonce_title.txt')
    text_title = loadfile('data/train/ann_annonce_title.txt')
    logging.info('load file ann_content.txt')
    text_content = loadfile('data/train/ann_content.txt')
    sentences = []
    for i in range(len(text_title)):
        if text_content[i] == 'NULL':
            sentences.append(text_title[i])
        else:
            sentences.append(text_title[i] + text_content[i])
    split_words_and_save(sentences, 'data/train/ann_title_content_split.txt')


if __name__ == "__main__":
    # 主程序
    logging.basicConfig(filename='split_logger.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #################################################################################################
    # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s : %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################
    # main()
    split_test_files('T1')
    # split_test_files('T2')
