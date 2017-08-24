# /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import csv
import codecs
import re
import os

# DATA_PATH=os.path.dirname(__file__) + "/data/train/"
# TEST_DATA_PATH = os.path.dirname(__file__) + "/data/test/T1/"
DATA_PATH="D:/iBaguo/work_home/bot/bot2017Fin-master/data/train/"
TEST_DATA_PATH = "D:/iBaguo/work_home/bot/bot2017Fin-master/data/test/T2/"
# files=['pricedetail.json','AnnouncementsRelations.json','AnnouncementsTrainSample.json','ResearchRelation.json','ResearchTrainSample.json']
# files={'price':'pricedetail.json', 'research_rel':'ResearchRelation.json', 'research_train':'ResearchTrainSample.json'}
# files={'price':'pricedetail.json', 'announce_rel':'AnnouncementsRelations.json', 'announce_train':'AnnouncementsTrainSample.json'}
data={}


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


def handle_utf8bom():
    files = ['pricedetail.json','AnnouncementsRelations.json','AnnouncementsTrainSample.json','ResearchRelation.json','ResearchTrainSample.json','NewsRelations.json', 'NewsTrainSample.json']
    for file in files:
        convert_utf8bom(DATA_PATH + file)


def json2csv(jsonfile, outfilename):
    with open(jsonfile, 'rb') as pf:
        load_data = json.load(pf)

    csvfile = open(outfilename,'w')
    fieldnames = ["news_id", "title", "annonce_type", "publish_date", "notice_date"]
    #fieldnames = ["news_id", "title", "column_type"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for val in load_data:
        for k in val.keys():
            if k == "content":
                del val[k]
                break
        writer.writerow(val)


def preload_data(pricefile, relationfile, trainfile):
    global data
    data.clear()
    files = {'price':pricefile, 'relation':relationfile, 'train':trainfile}
    for key in files:
        with codecs.open(DATA_PATH + files[key], 'r', 'utf-8') as filehandle:
            dict_str = filehandle.readlines()  # 读出来的是一个list
            jdict = json.loads(dict_str[0],encoding='utf-8')
            #jdict = byteify(jdict)
            data[key] = jdict
            print('file[%s]: %s loaded!' % (key, files[key]))
            # return print(file+'  loaded...!')


def pre_divide_research_data():
    preload_data('pricedetail.json','ResearchRelation.json', 'ResearchTrainSample.json')
    # research_files = {'price':'res_price_value.txt','news_id':'res_news_id.txt', 'title':'res_title.txt', 'column_type':'res_column_type.txt', 'content':'res_content.txt'}
    research_files = {'price':'res_price_value.txt'}
    for key in data['train'][0].keys():
        research_files[key] = 'res_' + key + '.txt'

    handles = {}
    for (key, file) in research_files.items():
        h = {}
        h['fd'] = codecs.open(DATA_PATH + file, 'w', 'utf-8')
        if key == 'price':
            h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=['value1', 'value2','value3'])
        else:
            h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=[key])
        handles[key] = h
    line_num = 0
    no_relation = 0
    for row in data['train']:
        line_num += 1
        news_id = row['news_id']
        relation = lookup_relation(news_id)
        if relation is None:
            no_relation += 1
            continue
        for k in research_files:
            # print('Key:', k, row)
            if k == 'content':
                if row.get('content') is None:
                    handles[k]['writer'].writerow({k: 'NULL'})
                else:
                    text = row['content'].strip(' \t\r\n\u3000')
                    # row['content'] = re.sub(r'[\r\n]', r'', row['content'])
                    handles[k]['writer'].writerow({k: re.sub('[\r\n\f\v\u007f\u001c\u001d\u001e\u001f]', '', text)})
            elif k == 'price':
                price = lookup_price(relation['security_code'], relation['publish_date'])
                if isinstance(price, dict):
                    handles['price']['writer'].writerow(price)
                else:
                    print('Missing Price %d: %s %s %s' % (line_num, news_id, relation['security_code'], relation['publish_date']))
            else:
                handles[k]['writer'].writerow({k:row[k]})
    for k in research_files:
        handles[k]['fd'].close()
    print('divide data done, no relation num: %d' % no_relation)


def pre_divide_announce_data():
    preload_data('pricedetail.json','AnnouncementsRelations.json', 'AnnouncementsTrainSample.json')
    # research_files = {'price':'res_price_value.txt','news_id':'res_news_id.txt', 'title':'res_title.txt', 'column_type':'res_column_type.txt', 'content':'res_content.txt'}
    # announce_files = {'price':'ann_price_value.txt','news_id':'ann_news_id.txt', 'annonce_title':'ann_annonce_title.txt', 'annonce_type':'ann_annonce_type.txt', 'content':'ann_content.txt'}
    announce_files = {'price':'ann_price_value.txt'}
    for key in data['train'][0].keys():
        announce_files[key] = 'ann_' + key + '.txt'

    handles = {}
    for (key, file) in announce_files.items():
        h = {}
        h['fd'] = codecs.open(DATA_PATH + file, 'w', 'utf-8')
        if key == 'price':
            h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=['value1', 'value2','value3'])
        else:
            h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=[key])
        handles[key] = h

    line_num = 0
    no_relation = 0
    for row in data['train']:
        line_num += 1
        news_id = row['news_id']
        relation = lookup_relation(news_id)
        if relation is None:
            no_relation += 1
            continue
        for k in announce_files:
            # print('Key:', k, row)
            if k == 'content':
                if row.get('content') is None:
                    handles[k]['writer'].writerow({k: 'NULL'})
                else:
                    text = row['content'].strip(' \t\r\n\u3000')
                    # row['content'] = re.sub(r'[\r\n]', r'', row['content'])
                    handles[k]['writer'].writerow({k: re.sub('[\r\n\f\v\u007f\u001c\u001d\u001e\u001f]', '', text)})
            elif k == 'price':
                price = lookup_price(relation['security_id'], relation['publish_date'])
                if isinstance(price, dict):
                    handles['price']['writer'].writerow(price)
                else:
                    print('Missing Price %d: %s %s %s' % (line_num, news_id, relation['security_id'], relation['publish_date']))
            else:
                handles[k]['writer'].writerow({k: row[k]})
    for k in announce_files:
        handles[k]['fd'].close()
    print('divide data done, no relation num: %d' % no_relation)



def pre_divide_news_data():
    preload_data('pricedetail.json','NewsRelations.json', 'NewsTrainSample.json')
    announce_files = {'price':'news_price_value.txt'}
    for key in data['train'][0].keys():
        announce_files[key] = 'news_' + key + '.txt'

    handles = {}
    for (key, file) in announce_files.items():
        h = {}
        h['fd'] = codecs.open(DATA_PATH + file, 'w', 'utf-8')
        if key == 'price':
            h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=['value1', 'value2','value3'])
        else:
            h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=[key])
        handles[key] = h
    line_num = 0
    no_relation = 0
    for row in data['train']:
        line_num += 1
        news_id = row['news_id']
        relation = lookup_relation(news_id)
        if relation is None:
            no_relation += 1
            continue
        for k in announce_files:
            # print('Key:', k, row)
            if k == 'content':
                if row.get('content') is None:
                    handles[k]['writer'].writerow({k: 'NULL'})
                else:
                    text = row['content'].strip(' \t\r\n')
                    # row['content'] = re.sub(r'[\r\n]', r'', row['content'])
                    text = re.sub('[\r\n\f\v\u007f\u001c\u001d\u001e\u001f]', '', text)
                    text = re.sub('<.+>', '', text)
                    handles[k]['writer'].writerow({k: text})
            elif k == 'price':
                price = lookup_price(relation['security_id'], relation['publish_date'])
                if isinstance(price, dict):
                    handles['price']['writer'].writerow(price)
                else:
                    print('Missing Price %d: %s %s %s' % (line_num, news_id, relation['security_id'], relation['publish_date']))
            else:
                handles[k]['writer'].writerow({k: row[k]})
    for k in announce_files:
        handles[k]['fd'].close()
    print('divide data done, no relation num: %d' % no_relation)



def lookup_relation(news_id):
    for row in data['relation']:
        if row['news_id'] == news_id:
            return row


def lookup_price(security_id, date):
    for row in data['price']:
        if row['security_id'] == security_id and row['data_date'] == date:
            return {"value1":row['d1_open']/row['d0_open']-1, "value2":row['d2_open']/row['d1_open']-1, "value3":row['d3_open']/row['d2_open']-1}


def pre_divide_test_data():
    files = ['AnnouncementsTrainSample.json', 'ResearchTrainSample.json', 'NewsTrainSample.json']
    data = []
    uuid = []
    for file in files:
        data.clear()
        convert_utf8bom(TEST_DATA_PATH + file)
        with codecs.open(TEST_DATA_PATH + file, 'r', 'utf-8') as filehandle:
            dict_str = filehandle.readlines()  # 读出来的是一个list
            data = json.loads(dict_str[0], encoding='utf-8')
            print('file: %s loaded!' % file)
        announce_files = {}
        prefix = file[0:3] + '_'
        for key in data[0].keys():
            announce_files[key] = prefix + key + '.txt'

        handles = {}
        for (key, file) in announce_files.items():
            h = {}
            h['fd'] = codecs.open(TEST_DATA_PATH + file, 'w', 'utf-8')
            if key == 'price':
                h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=['value1', 'value2','value3'])
            else:
                h['writer'] = csv.DictWriter(h['fd'], delimiter='\t', fieldnames=[key])
            handles[key] = h

        line_num = 0
        for line in data:
            line_num += 1
            for k in announce_files.keys():
                if k == 'content':
                    if line.get('content') is None:
                        handles[k]['writer'].writerow({k: 'NULL'})
                    else:
                        text = line['content'].strip(' \t\r\n\u3000')
                        text = re.sub('[ \t\r\n\f\v\u007f\u001c\u001d\u001e\u001f]', '', text)
                        if file[0:3] == 'New':
                            text = re.sub('<[^<>]>', '', text)
                        handles[k]['writer'].writerow({k: text})
                elif k == 'Content':
                    if line.get('Content') is None:
                        handles[k]['writer'].writerow({k: 'NULL'})
                    else:
                        text = line['Content'].strip(' \t\r\n\u3000')
                        text = re.sub('[ \t\r\n\f\v\u007f\u001c\u001d\u001e\u001f]', '', text)
                        if file[0:3] == 'New':
                            text = re.sub('<[^<>]>', '', text)
                        handles[k]['writer'].writerow({k: text})
                else:
                    handles[k]['writer'].writerow({k: line[k]})
                if k == 'uuid' and line[k] not in uuid:
                    uuid.append(line[k])
        for k in announce_files:
            handles[k]['fd'].close()
        print('divide data done')
    with codecs.open(TEST_DATA_PATH + 'pred_uuid.txt', 'w', 'utf-8') as filehandle:
        for line in uuid:
            filehandle.write(line + '\n')



if __name__ == '__main__':
    # handle_utf8bom()
    #json2csv('ResearchTrainSample.json', 'ResearchTrainSample.csv')
    # json2csv('AnnouncementsTrainSample.json', 'AnnouncementsTrainSample.csv')
    # tr.handel_utf8bom()
    # pre_divide_research_data()
    # pre_divide_announce_data()
    # pre_divide_news_data()
    #
    pre_divide_test_data()








