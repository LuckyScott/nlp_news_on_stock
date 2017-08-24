# /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import codecs
import csv
import re
import numpy as np
import os

DATA_PATH="D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\data\\train\\"
TEST_DATA_PATH = "D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\"

data = {}

def load_data(doc_wordfile, corpus_file):
    global data
    with codecs.open(corpus_file, 'r', 'utf-8') as fd:
        lines = fd.readlines()
        corpus = []
        for line in lines:
            # line = line.rstrip(' \t\r\n')
            corpus.append(json.loads(line, encoding='utf-8'))
        data['corpus'] = corpus

    with codecs.open(doc_wordfile, 'r', 'utf-8') as fd:
        lines = fd.readlines()
        docs = []
        for line in lines:
            docs.append(json.loads(line,encoding='utf-8'))
        data['docs'] = docs

    # calc idf
    doc_num = len(docs)
    idf_data = calc_idf(data['corpus'], doc_num)
    filename = os.path.basename(corpus_file)
    save_idf(idf_data, os.path.splitext(filename)[0] + '_idf_val.txt')
    data['idf_data'] = idf_data
    print('load data done!')


def calc_idf(corpus, docnum):
    idf_data = []
    low_freq_num = 0
    for dic in corpus:
        idf_dict = {}
        idf_dict['word_type'] = dic['word_type']
        idf_dict['idf'] = {}
        for (key, val) in dic['corpus'].items():
            # 过虑掉低频词
            if val['freq'] < 10:
                low_freq_num += 1
                continue
            idf_dict['idf'][key] = np.log(docnum/val['freq'])
        idf_data.append(idf_dict)
    print('low freq num:%d' % low_freq_num)
    return idf_data


def save_idf(idf_data, filename):
    with codecs.open(filename, 'w', encoding='utf-8') as fd:
        # 一行一行写
        for row in idf_data:
            fd.write(json.dumps(row, ensure_ascii=False) + '\n')


def calc_tfidf(sparse=False):
    tfidf_matrix = []
    i = 0
    # max_row = 0
    max_column = 0
    for doc in data['docs']:
        tfidf_row = []
        i += 1
        j = 0
        for row in data['idf_data']:
            for (key, idf) in row['idf'].items():
                j += 1
                if key in doc['words']:
                    tf = doc['words'][key] / doc['total_words']
                    if sparse:
                        tfidf_matrix.append((i-1, j-1, tf*idf))
                    else:
                        # tfidf_row.append(round(tf*idf,6))
                        tfidf_row.append(tf*idf)
                else:
                    if not sparse:
                        tfidf_row.append(0)
        if max_column < j:
            max_column = j
        if not sparse:
            tfidf_matrix.append(tfidf_row)
    max_row = i
    fieldnames = []
    for row in data['idf_data']:
        for i in range(len(row['idf'])):
            fieldnames.append(row['word_type'] + '_x' + str(i+1))
    tfidf = {'isSparse':sparse, 'size':(max_row, max_column),'fields': fieldnames, 'data': tfidf_matrix}
    # if sparse:
    #     tfidf = {'fields': fieldnames, 'data': tfidf_matrix}
    # else:
    #     tfidf = {'fields':fieldnames, 'data':np.asarray(tfidf_matrix)}
    return tfidf


def save_tfidf(data, filename):
    '''
    :param data: numpy.dnarray, tfidf matrix 
    :param filename: string,  output file name
    :return: None
    '''
    # fd = codecs.open(filename, 'w', encoding='utf-8')
    # for r in data:
    #     fd.write(json.dumps(r.tolist(), ensure_ascii=False) + '\n')
    # fd.close()

    csvfile = codecs.open(filename, 'w', encoding='utf-8')
    fieldnames = data['fields']
    # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    if data['isSparse']:
        row = []
        ii = 0
        jj = 0
        for val in data['data']:
            if jj >= data['size'][1]:
                writer.writerow(row)
                row.clear()
                ii += 1
                jj = 0
            while ii < val[0] and ii < data['size'][0]:
                while jj < data['size'][1]:
                    row.append(0)
                    jj += 1
                writer.writerow(row)
                row.clear()
                ii += 1
                jj = 0
            if ii == val[0]:
                while jj < val[1]:
                    row.append(0)
                    jj += 1
                if jj == val[1]:
                    row.append(val[2])
                    jj += 1
                else:
                    print('data error 1')
            else:
                print('Data Error 2')
    else:
        for val in data['data']:
            writer.writerow(val)
    csvfile.close()


def generate_weka_arff(tfidf_data, classify_file, outfile):
    classify = []
    with codecs.open(classify_file, 'r', encoding='utf-8') as fd:
        reader = csv.reader(fd)
        for row in reader:
            classify.append(row)
    # if len(classify) != len(tfidf_data['data']):
    #     raise Exception('Value error: classify data rows not match tfidf data rows. classify file: %s' % classify_file)
    header = '''
@ATTRIBUTE classA_1day 	{1,2,3}
@ATTRIBUTE classA_2day 	{1,2,3}
@ATTRIBUTE classA_3day 	{1,2,3}
@ATTRIBUTE classB_1day 	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
@ATTRIBUTE classB_2day 	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
@ATTRIBUTE classB_3day 	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
'''
    arff = codecs.open(outfile,'w', encoding='utf-8')
    arff.write('@RELATION ' + os.path.splitext(os.path.basename(outfile))[0] + '\n\n')
    arff.write(header)
    fields = tfidf_data['fields']
    for field in fields:
        arff.write('@ATTRIBUTE ' + field + ' REAL\n')
    arff.write('\n\n@DATA\n')
    if tfidf_data['isSparse']:
        ii = 0
        jj = 0
        for val in tfidf_data['data']:
            if jj >= tfidf_data['size'][1]:
                arff.write('\n')
                ii += 1
                jj = 0
            while ii < val[0] and ii < tfidf_data['size'][0]:
                if jj == 0:
                    for j in range(len(classify[ii])):
                        if j == 0:
                            arff.write(classify[ii][j])
                        else:
                            arff.write(',' + str(classify[ii][j]))
                while jj < tfidf_data['size'][1]:
                    arff.write(',0')
                    jj += 1
                arff.write('\n')
                ii += 1
                jj = 0
            if ii == val[0]:
                if jj == 0:
                    for j in range(len(classify[ii])):
                        if j == 0:
                            arff.write(classify[ii][j])
                        else:
                            arff.write(',' + str(classify[ii][j]))
                while jj < val[1]:
                    arff.write(',0')
                    jj += 1
                if jj == val[1]:
                    arff.write(',' + str(val[2]))
                    jj += 1
                else:
                    print('data error 1')
            else:
                print('Data Error 2')
        while ii < tfidf_data['size'][0]:
            if jj == 0:
                for j in range(len(classify[ii])):
                    if j == 0:
                        arff.write(classify[ii][j])
                    else:
                        arff.write(',' + str(classify[ii][j]))
            while jj < tfidf_data['size'][1]:
                arff.write(',0')
                jj += 1
            arff.write('\n')
            ii += 1
            jj = 0
    else:
        for i in range(len(classify)):
            # row = ','.join(str(x) for x in classify[i]) + ',' + ','.join(str(x) for x in tfidf_data['data'][i])
            # arff.write(row + '\n')
            for j in range(len(classify[i])):
                if j == 0:
                    arff.write(classify[i][j])
                else:
                    arff.write(',' + str(classify[i][j]))
            n = len(tfidf_data['data'][i])
            for j in range(n):
                arff.write( ',' + str(tfidf_data['data'][i][j]))
            arff.write('\n')
    arff.close()


def generate_test_weka_arff(tfidf_data, outfile):
    header = '''
@ATTRIBUTE classA_1day 	{1,2,3}
@ATTRIBUTE classA_2day 	{1,2,3}
@ATTRIBUTE classA_3day 	{1,2,3}
@ATTRIBUTE classB_1day 	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
@ATTRIBUTE classB_2day 	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
@ATTRIBUTE classB_3day 	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
'''
    arff = codecs.open(outfile,'w', encoding='utf-8')
    arff.write('@RELATION ' + os.path.splitext(os.path.basename(outfile))[0] + '\n\n')
    arff.write(header)
    fields = tfidf_data['fields']
    for field in fields:
        arff.write('@ATTRIBUTE ' + field + ' REAL\n')
    arff.write('\n\n@DATA\n')
    max_row, max_column = tfidf_data['size']
    if tfidf_data['isSparse']:
        ii = 0
        jj = 0
        for val in tfidf_data['data']:
            if jj >= max_column:
                arff.write('\n')
                ii += 1
                jj = 0
            while ii < val[0] and ii < max_row:
                if jj == 0:
                    for j in range(6):
                        if j == 0:
                            arff.write('?')
                        else:
                            arff.write(',?')
                while jj < max_column:
                    arff.write(',0')
                    jj += 1
                arff.write('\n')
                ii += 1
                jj = 0
            if ii == val[0]:
                if jj == 0:
                    for j in range(6):
                        if j == 0:
                            arff.write('?')
                        else:
                            arff.write(',?')
                while jj < val[1]:
                    arff.write(',0')
                    jj += 1
                if jj == val[1]:
                    arff.write(',' + str(val[2]))
                    jj += 1
                else:
                    print('data error 1')
            else:
                print('Data Error 2')
        while ii < max_row:
            if jj == 0:
                for j in range(6):
                    if j == 0:
                        arff.write('?')
                    else:
                        arff.write(',?')
            while jj < max_column:
                arff.write(',0')
                jj += 1
            arff.write('\n')
            ii += 1
            jj = 0
    else:
        for i in range(max_row):
            # row = ','.join(str(x) for x in classify[i]) + ',' + ','.join(str(x) for x in tfidf_data['data'][i])
            # arff.write(row + '\n')
            for j in range(6):
                if j == 0:
                    arff.write('?')
                else:
                    arff.write(',?')
            for j in range(max_column):
                arff.write( ',' + str(tfidf_data['data'][i][j]))
            arff.write('\n')
    arff.close()


def main():
    files = [
            {
                'docfile': 'res_title_word_seg_pos_freq.txt',
                'words_file': 'res_title_word_seg_pos_idf_freq_classify.txt',
                'tfidf': 'res_title_word_seg_pos_tfidf.csv',
                'classify': 'res_price_classify.txt',
                'arff':'res_title.arff'
            },
            {
                'docfile': 'res_content_word_seg_pos_freq.txt',
                'words_file': 'res_content_word_seg_pos_idf_freq_classify.txt',
                'tfidf': 'res_content_word_seg_pos_tfidf.csv',
                'classify': 'res_price_classify.txt',
                'arff':'res_content.arff'
            },
            {
                'docfile': 'ann_title_word_seg_pos_freq.txt',
                'words_file': 'ann_title_word_seg_pos_idf_freq_classify.txt',
                'tfidf': 'ann_title_word_seg_pos_tfidf.csv',
                'classify': 'ann_price_classify.txt',
                'arff':'ann_title.arff'
            },
            {
                'docfile': 'ann_content_word_seg_pos_freq.txt',
                'words_file': 'ann_content_word_seg_pos_idf_freq_classify.txt',
                'tfidf': 'ann_content_word_seg_pos_tfidf.csv',
                'classify': 'ann_price_classify.txt',
                'arff': 'ann_content.arff'
            }
            ]
    for file in files:
        load_data(DATA_PATH + file['docfile'], DATA_PATH + file['words_file'])
        tfidf = calc_tfidf(True)
        save_tfidf(tfidf, DATA_PATH + file['tfidf'])
        generate_weka_arff(tfidf, DATA_PATH + file['classify'], DATA_PATH + file['arff'])


def test_main():
    files = [
        {
            'docfile': 'res_title_word_seg_pos_freq.txt',
            'words_file': 'res_title_word_seg_pos_idf_freq_classify.txt',
            'tfidf': 'res_title_word_seg_pos_tfidf.csv',
            # 'classify': 'res_price_classify.txt',
            'arff': 'res_title.arff'
        },
        {
            'docfile': 'res_content_word_seg_pos_freq.txt',
            'words_file': 'res_content_word_seg_pos_idf_freq_classify.txt',
            'tfidf': 'res_content_word_seg_pos_tfidf.csv',
            # 'classify': 'res_price_classify.txt',
            'arff': 'res_content.arff'
        },
        {
            'docfile': 'ann_title_word_seg_pos_freq.txt',
            'words_file': 'ann_title_word_seg_pos_idf_freq_classify.txt',
            'tfidf': 'ann_title_word_seg_pos_tfidf.csv',
            # 'classify': 'ann_price_classify.txt',
            'arff': 'ann_title.arff'
        },
        {
            'docfile': 'ann_content_word_seg_pos_freq.txt',
            'words_file': 'ann_content_word_seg_pos_idf_freq_classify.txt',
            'tfidf': 'ann_content_word_seg_pos_tfidf.csv',
            # 'classify': 'ann_price_classify.txt',
            'arff': 'ann_content.arff'
        }
    ]
    for file in files:
        load_data(TEST_DATA_PATH + file['docfile'], DATA_PATH + file['words_file'])
        tfidf = calc_tfidf()
        save_tfidf(tfidf, TEST_DATA_PATH + file['tfidf'])
        generate_test_weka_arff(tfidf, TEST_DATA_PATH + file['arff'])



if __name__ == '__main__':
    # main()
    test_main()