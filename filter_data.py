# /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import transform as tr
import codecs
import re
import os

DATA_PATH="D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\data\\train\\"
TEST_DATA_PATH = "D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\"

def filter_empty_lines(price_file, text_infile, text_outfile):
    with codecs.open(price_file, 'r', 'utf-8') as fd:
        price_lines = fd.readlines()
    with codecs.open(text_infile, 'r', 'utf-8') as fd:
        data_lines = fd.readlines()
    out = codecs.open(text_outfile, 'w', 'utf-8')
    # new_data = []
    line_num = 0
    new_line = 0
    for line in price_lines:
        line_num += 1
        if line[0] != 'N':
            new_line += 1
            row = data_lines[line_num - 1].strip(' []\t\r\n') + '\n'
            out.write(row)
        else:
            pass
            # print('line:%d' % line_num, data_lines[line_num-1])
    out.close()
    print('%s is done, total lines:%d -->%d!' % (text_infile, line_num, new_line))


def filter_invalid_words(infile, outfile):
    with codecs.open(infile, 'r', 'utf-8') as fd:
        lines = fd.readlines()
    # fd = codecs.open(infile, 'r', 'utf-8')
    out = codecs.open(outfile, 'w', 'utf-8')
    total_line = len(lines)
    line_num = 0
    filter_chars = ['\u3000','·','~','!','@','#','$','%','^','&','*','(',')','-','_','+','=','[',']','{','}','|','\\',';',':','"','\'','<','>',',','.','?','/']
    # last_line = ''
    for l in lines:
        line_num += 1
        line = l.strip(' \t\r\n')
        if line == '':
            print('line: %d is empty: 0x%s' % (line_num, l.encode('utf-8').hex()))
            # out.write('null\n')
            continue
        # if line[-1] != ']':
        #     last_line += line
        #     continue
        # else:
        #     if last_line != '':
        #         line = last_line + line
        #         last_line = ''
        if line[0] != '[':
            # print('line: %d is invalid line, content: %s(%s)' % (line_num, line, line.encode('utf-8').hex()))
            print('line: %d is invalid line, content: %s' % (line_num, line))
            continue
        line = line.strip('[]')
        row = line.split(', ')
        new_row = []
        for w in row:
            w = w.strip(' \t')
            # 过虑掉没有标记词性的
            rws = re.match('^(.+)/(\w+)$', w)
            if rws is None:
                continue
            rws = rws.groups()
            rw = rws[1]
            # 去掉标点
            # if re.match('^[w]',rws[1]):
            if rw[0] == 'w':
                continue
            rw = rws[0]
            if re.match('^\d+[.]{0,1}\d*%', rw):
                continue
            elif rw in filter_chars:
                continue
            elif re.match('^[ \ta-zA-Z/<>(),.;:，。“”‘’\'\"：；《》】【！（）?？、_\-=—\u3000\u007f]*$', rw):
                continue
            new_row.append(w)
        if len(new_row) == 0:
            # print('line: %d is empty: %s'% (line_num, line))
            out.write('null\n')
        else:
            out.write(', '.join(new_row) + '\n')
        if line_num % 3000 == 0:
            print('at %d of %d(%f)%%' % (line_num, total_line, line_num/total_line*100))
    out.close()
    print('%s filter done.' % infile)


def calc_doc_word_freqs(infile, outfile):
    with codecs.open(infile, 'r', 'utf-8') as fd:
        lines = fd.readlines()
    out = codecs.open(outfile, 'w', 'utf-8')
    line_num = 0
    for line in lines:
        line_num += 1
        line = line.rstrip(' \t\r\n')
        freq_row = {}
        #空行
        if line == 'null':
            write_row = {'total_words': 0, 'word_count': 0, 'words': {}}
            out.write(json.dumps(write_row, ensure_ascii=False) + '\n')
            continue
        row = line.split(',')
        for w in row:
            w = w.strip()
            if w in freq_row:
                freq_row[w] += 1
            else:
                freq_row[w] = 1
        write_row = {'total_words':len(row), 'word_count':len(freq_row),'words':freq_row}
        out.write(json.dumps(write_row, ensure_ascii=False) + '\n')
    out.close()
    print('%s load done.' % infile)


def get_sort_key(x):
    try:
        res = re.search(r'/(\w+)$',x[0]).groups()
        return res[0]
    except Exception as err:
        print(x)


def get_word_pos(word):
    try:
        res = re.match(r'^(.+)\s*/(\w+)$',word).groups()
        return res[1]
    except Exception as err:
        print(word)


def calc_word_idf_freqs(filename):
    with codecs.open(filename, 'r', 'utf-8') as fd:
        lines = fd.readlines()
    word_list = {}
    for line in lines:
        json_data = json.loads(line, encoding='utf-8')
        for word in json_data['words']:
            if word in word_list:
                word_list[word]['freq'] += 1
            else:
                word_list[word] = {'pos':get_word_pos(word), 'freq':1}
    # 按词性排序
    # sorted_list = dict(sorted(word_list.items(), key=lambda x: re.search(r'/(\w+)$',x[0]).groups()[0], reverse=False))
    # sorted_list = dict(sorted(word_list.items(), key=get_sort_key, reverse=False))
    sorted_list = dict(sorted(word_list.items(), key=lambda x: x[1]['pos'], reverse=False))
    data = {'word_count':len(sorted_list), 'corpus':sorted_list}
    print('%s calc done, word_count:%d.' % (filename, len(sorted_list)))
    return data


def save_word_idf_freqs(data,save_file):
    with codecs.open(save_file, 'w', 'utf-8') as fd:
        fd.write(json.dumps(data, ensure_ascii=False))


# def do_filter_empty_lines():
#     files = [
#              {'price_file':'res_price_value.txt','in': 'res_title_seg_pos_utf8.txt', 'out': 'res_title_valid_line_seg_pos.txt'},
#              {'price_file':'res_price_value.txt','in': 'res_content_seg_pos_utf8.txt', 'out': 'res_content_valid_line_seg_pos.txt'},
#              {'price_file':'ann_price_value.txt','in': 'ann_title_seg_pos_utf8.txt', 'out': 'ann_title_valid_line_seg_pos.txt'},
#              {'price_file':'ann_price_value.txt','in': 'ann_content_seg_pos_utf8.txt', 'out': 'ann_content_valid_line_seg_pos.txt'}
#             ]
#     for file in files:
#         filter_empty_lines(file['price_file'], file['in'], file['out'])


def do_filter_invalide_words():
    files = [
             {'in': 'res_title_seg_pos.txt', 'out': 'res_title_valid_word_seg_pos.txt'},
             {'in': 'res_content_seg_pos.txt', 'out': 'res_content_valid_word_seg_pos.txt'},
             {'in': 'ann_annonce_title_seg_pos.txt', 'out': 'ann_title_valid_word_seg_pos.txt'},
             {'in': 'ann_content_seg_pos.txt', 'out': 'ann_content_valid_word_seg_pos.txt'}
            ]
    for file in files:
        filter_invalid_words(DATA_PATH + file['in'], DATA_PATH + file['out'])


def do_calc_doc_freqs():
    files = [
             {'in':'res_title_valid_word_seg_pos.txt', 'out':'res_title_word_seg_pos_freq.txt'},
             {'in': 'res_content_valid_word_seg_pos.txt', 'out': 'res_content_word_seg_pos_freq.txt'},
             {'in': 'ann_title_valid_word_seg_pos.txt', 'out': 'ann_title_word_seg_pos_freq.txt'},
             {'in': 'ann_content_valid_word_seg_pos.txt', 'out': 'ann_content_word_seg_pos_freq.txt'}
            ]
    for file in files:
        calc_doc_word_freqs(DATA_PATH + file['in'], DATA_PATH + file['out'])


def do_calc_word_idf_freqs():
    files = [
             {'in':'res_title_word_seg_pos_freq.txt', 'out':'res_title_word_seg_pos_idf_freq.txt'},
             {'in': 'res_content_word_seg_pos_freq.txt', 'out': 'res_content_word_seg_pos_idf_freq.txt'},
             {'in': 'ann_title_word_seg_pos_freq.txt', 'out': 'ann_title_word_seg_pos_idf_freq.txt'},
             {'in': 'ann_content_word_seg_pos_freq.txt', 'out': 'ann_content_word_seg_pos_idf_freq.txt'}
            ]
    for file in files:
        data = calc_word_idf_freqs(DATA_PATH + file['in'])
        save_word_idf_freqs(data, DATA_PATH + file['out'])
        classify_word(data, os.path.splitext(os.path.basename(file['out']))[0])


def classify_word(data, file_prefix):
    # data = {'word_count': len(sorted_list), 'words': sorted_list}
    # i: 成语, s: 方位, b: 区别词（惯用语）, j:术语?(如：质检，冬奥会，工商联), l:术语(4字，如：人民法院，固定资产)
    # types = {'n':'noun', 'v':'verb', 'a':'adj','j':'j','l':'l','s':'s','t':'time', 'b':'b', 'i':'i'}
    # 分 3 个文件，名词、形容词、动词
    noun_words = {'word_type':'noun','word_count':0,'corpus':{}}
    verb_words = {'word_type':'verb','word_count':0,'corpus':{}}
    adj_words = {'word_type':'adj','word_count':0,'corpus':{}}
    for (key, val) in data['corpus'].items():
        if val['pos'][0] in ['n','j','l','i','t']:
            noun_words['corpus'][key] = val
        elif val['pos'][0] == 'v':
            verb_words['corpus'][key] = val
        elif val['pos'][0] in ['a','s','b']:
            adj_words['corpus'][key] = val
        else:
            # 其他词性不用
            pass
    noun_words['word_count'] = len(noun_words['corpus'])
    verb_words['word_count'] = len(verb_words['corpus'])
    adj_words['word_count'] = len(adj_words['corpus'])
    with codecs.open(DATA_PATH + file_prefix + '_classify.txt', 'w', 'utf-8') as fd:
        fd.write(json.dumps(noun_words, ensure_ascii=False) + '\n')
        fd.write(json.dumps(verb_words, ensure_ascii=False) + '\n')
        fd.write(json.dumps(adj_words, ensure_ascii=False) + '\n')


def do_filter_test_invalide_words():
    files = [
             {'in': 'Res_title_seg_pos.txt', 'out': 'res_title_valid_word_seg_pos.txt'},
             {'in': 'Res_content_seg_pos.txt', 'out': 'res_content_valid_word_seg_pos.txt'},
             {'in': 'Ann_AnnonceTitle_seg_pos.txt', 'out': 'ann_title_valid_word_seg_pos.txt'},
             {'in': 'Ann_Content_seg_pos.txt', 'out': 'ann_content_valid_word_seg_pos.txt'}
            ]
    for file in files:
        filter_invalid_words(TEST_DATA_PATH + file['in'], TEST_DATA_PATH + file['out'])


def do_calc_test_doc_freqs():
    files = [
             {'in':'res_title_valid_word_seg_pos.txt', 'out':'res_title_word_seg_pos_freq.txt'},
             {'in': 'res_content_valid_word_seg_pos.txt', 'out': 'res_content_word_seg_pos_freq.txt'},
             {'in': 'ann_title_valid_word_seg_pos.txt', 'out': 'ann_title_word_seg_pos_freq.txt'},
             {'in': 'ann_content_valid_word_seg_pos.txt', 'out': 'ann_content_word_seg_pos_freq.txt'}
            ]
    for file in files:
        calc_doc_word_freqs(TEST_DATA_PATH + file['in'], TEST_DATA_PATH + file['out'])


def main():
    # do_filter_empty_lines()
    do_filter_invalide_words()
    do_calc_doc_freqs()
    do_calc_word_idf_freqs()

def test_main():
    do_filter_test_invalide_words()
    do_calc_test_doc_freqs()


if __name__ == '__main__':
    # main()
    test_main()





