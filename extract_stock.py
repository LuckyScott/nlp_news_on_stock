import os
import json
import codecs
import re
import tushare as ts
import datetime
import csv
import time
import random
from urllib.request import urlopen, Request

from scoring import *

price_detail = {}
data = {}
DATA_PATH='D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\'

def load_price_detail(file):
    global  price_detail
    price_detail = {}
    print('load %s from %s' % (file, DATA_PATH))
    with codecs.open(DATA_PATH + file, 'r', encoding='utf-8') as fd:
        dict_str = fd.read()
        json_data = json.loads(dict_str, encoding='utf-8')
        for r in json_data:
            uuid = r['uuid']
            del r['uuid']
            price_detail[uuid] = r
    print('load price detail from {} success!'.format(file))


def parser_ch_date(year, mon, day):
    years = {'一零':2010, '一\u3007':2010,'一O':2010, '一o':2010, '一一':2011,'一二':2012,'一三':2013, '一四': 2014,'一五':2015,'一六':2016,'一七':2017}
    months = {'一': 1, '二':2, '三':3,'四': 4,'五': 5, '六':6,'七':7,'八':8,'九':9,'十':10,'十一':11,'十二':12}
    days = {'一': 1, '二':2, '三':3,'四': 4,'五': 5, '六':6,'七':7,'八':8,'九':9,'十':10,
            '一十一': 11, '一十二': 12, '一十三': 13, '一十四': 14, '一十五': 15, '一十六': 16, '一十七': 17, '一十八': 18, '一十九': 19,
            '十一':11,'十二':12,'十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
            '二十一': 21, '二十二': 22,'二十三': 23, '二十四': 24, '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29, '三十': 30,
            '三十一': 31
            }
    return datetime.date(years[year], months[mon], days[day]).isoformat()


def load_announce_data(file):
    with codecs.open(DATA_PATH + file, 'r', 'utf-8') as filehandle:
        dict_str = filehandle.read()  # 读出来的是一个list
        jdict = json.loads(dict_str,encoding='utf-8')
        print(len(jdict))
    data = []
    for r in jdict:
        uuid = r['uuid']
        if uuid in price_detail:
            date = r['news_id'][2:6] + '-' + r['news_id'][6:8] + '-' + r['news_id'][8:10]
            if r.get('content') is not None:
                content = r['content'][-260:].strip(' \r\n')
                ma = re.match('.+(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日([\s\r\n]*(备查.+|报备.+|附.+|於本(公告|通知).+|[\-－\s\r\n\d]+|第.+页))?$', content, re.DOTALL)
                if ma is not None:
                    g = ma.groups()[:3]
                    date = datetime.date(*[ int(x) for x in g]).isoformat()
                    # print('Date:', date, 'uuid:', uuid)
                else:
                    ma = re.match('.+\s*二.(一[一二三四五六七])年([十]?[一二三四五六七八九十])月([一二三]?[十]?[一二三四五六七八九十])日([\s\r\n]*(备查.+|报备.+|附.+|於本(公告|通知).+|[\-－\s\r\n\d]+|第.+页))?$', content, re.DOTALL)
                    if ma is not None:
                        date = parser_ch_date(*ma.groups()[:3])
                    else:
                        pass
                        # print('Content:' ,content, [hex(ord(x)) for x in content[-10:]])
            title = r['annonce_title']
            ma = re.match('^((00|30|60)\d{4})', title)
            code = None
            if ma is None:
                if r.get('content') is not None:
                    content = r['content']
                    ma = re.search('(证券代码|股票代码)：((00|30|60)\d{4})', content)
                    if ma is not None:
                        code = ma.groups()[1].strip()
            else:
                code = ma.groups()[0].strip()
            if code is not None:
                data.append({'uuid': uuid, 'date': date, 'stock_code': code})
    print('valid line: %d' % len(data))
    print('check valid date...')
    same_num = 0
    tmp_data = []
    for row in data:
        day = datetime.date(*[int(x) for x in row['date'].split('-')])
        wday = day.weekday()
        if wday + 2 == price_detail[row['uuid']]['d0_wd']:
            same_num += 1
            # print(row, 'not  same', day, wday, price_detail[row['uuid']])
            tmp_data.append(row)
        else:
            pass
            # print(row, 'not  same', day, wday, price_detail[row['uuid']])
    print('same day num = %d ' % same_num)
    data = tmp_data
    tmp_data = []
    tmp_uuid = {}
    for i in range(len(data)):
        d = data[i]
        if d['uuid'] not in tmp_uuid:
            tmp_uuid[d['uuid']] = 0
        else:
            continue
        year, mon, day = [int(x) for x in d['date'].split('-')]
        start_day = datetime.date(year, mon, day)
        end_day = datetime.date(year, mon, day) + datetime.timedelta(days = 10)
        try:
            kline = get_stock_kline_data(d['stock_code'], start_day.isoformat(), end_day.isoformat(), 'qfq')
            if kline is None:
                continue
            if kline[0][0] == d['date']:
                r = data[i]
                r['open0'] = float(kline[0][1])
                r['open1'] = float(kline[1][1])
                r['open2'] = float(kline[2][1])
                r['open3'] = float(kline[3][1])
                tmp_data.append(r)
            else:
                print('date not match', d, kline[0])
        except Exception:
            print(d)
        time.sleep(0.2)
    data = tmp_data
    with codecs.open(DATA_PATH + 'announce_stock_code_price.txt', 'w', 'utf-8') as fd:
        writer = csv.DictWriter(fd, delimiter='\t', fieldnames=list(data[0].keys()))
        writer.writeheader()
        for r in data:
            writer.writerow(r)
    return data

def load_research_data(file):
    with codecs.open(DATA_PATH + file, 'r', 'utf-8') as filehandle:
        dict_str = filehandle.read()  # 读出来的是一个list
        jdict = json.loads(dict_str,encoding='utf-8')
        print(len(jdict))
    data = []
    for r in jdict:
        uuid = r['uuid']
        if uuid in price_detail:
            date = r['news_id'][2:6] + '-' + r['news_id'][6:8] + '-' + r['news_id'][8:10]
            title = r['title']
            code = None
            if r.get('content') is not None:
                content = r['content']
                ma = re.match('^.+?(\(|（)((00|30|60)\d{4})', content)
                if ma is not None:
                    code = ma.groups()[1].strip()
            if code is not None:
                data.append({'uuid': uuid, 'date': date, 'stock_code': code})
    print('valid line: %d' % len(data))
    print('check valid date...')
    same_num = 0
    tmp_data = []
    for row in data:
        day = datetime.date(*[int(x) for x in row['date'].split('-')])
        wday = day.weekday()
        if wday + 2 == price_detail[row['uuid']]['d0_wd']:
            same_num += 1
            tmp_data.append(row)
        else:
            pass
            # print(row, 'not  same', day, wday, price_detail[row['uuid']])
    print('same day num = %d ' % same_num)
    data = tmp_data
    # get price
    tmp_data = []
    tmp_uuid = {}
    for i in range(len(data)):
        d = data[i]
        if d['uuid'] not in tmp_uuid:
            tmp_uuid[d['uuid']] = 0
        else:
            continue
        year, mon, day = [int(x) for x in d['date'].split('-')]
        start_day = datetime.date(year, mon, day)
        end_day = datetime.date(year, mon, day) + datetime.timedelta(days = 10)
        try:
            kline = get_stock_kline_data(d['stock_code'], start_day.isoformat(), end_day.isoformat(), 'qfq')
            if kline is None:
                continue
            if kline[0][0] == d['date']:
                r = data[i]
                r['open0'] = float(kline[0][1])
                r['open1'] = float(kline[1][1])
                r['open2'] = float(kline[2][1])
                r['open3'] = float(kline[3][1])
                tmp_data.append(r)
            else:
                print('date not match', d, kline[0])
        except Exception:
            print(d)
        time.sleep(0.2)
    data = tmp_data
    with codecs.open(DATA_PATH + 'research_stock_code_price.txt', 'w', 'utf-8') as fd:
        writer = csv.DictWriter(fd, delimiter='\t', fieldnames=list(data[0].keys()))
        writer.writeheader()
        for r in data:
            writer.writerow(r)
    return data


def get_stock_kline_data(code, start='', end='', fq=''):
    code = 'sh%s' % code if code[:1] in ['5', '6', '9'] else 'sz%s' % code
    url = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=&param={},day,{},{},20,{}&r={}'.format(code,start, end, fq, random.random())
    request = Request(url)
    text = urlopen(request, timeout=10).read()
    data = json.loads(text.decode('utf-8'), encoding='utf-8')
    try:
        if fq == 'qfq':
            data = data['data'][code]['qfqday']
        elif fq == 'hfq':
            data = data['data'][code]['hfqday']
        else:
            data = data['data'][code]['day']
    except Exception:
        print(url)
        print(data)
        return None
    return data


def save_predict_value(result_value, filename):
    with codecs.open(filename, 'w', encoding='utf-8') as fd:
        csv_writer = csv.DictWriter(fd, delimiter='\t', fieldnames=['uuid', 'value1', 'value2','value3'])
        csv_writer.writeheader()
        for row in result_value:
            csv_writer.writerow(row)

def gen_result(file1, file2):
    predict = submitfile2listofdict(DATA_PATH + file1)
    price_result = submitfile2listofdict(DATA_PATH + file2)
    price = {}
    for r in price_result:
        uuid = r['uuid']
        del r['uuid']
        price[uuid] = r
    for i in range(len(predict)):
        row = predict[i]
        uuid = row['uuid']
        if uuid in price:
            predict[i]['value1'] = float(price[uuid]['open1']) / float(price[uuid]['open0']) - 1
            predict[i]['value2'] = float(price[uuid]['open2']) / float(price[uuid]['open1']) - 1
            predict[i]['value3'] = float(price[uuid]['open3']) / float(price[uuid]['open2']) - 1
    save_predict_value(predict, DATA_PATH + 'new_pred_result.txt')


def merge_data(res_data, ann_data):
    data = res_data + ann_data
    with codecs.open(DATA_PATH + 'price_result.txt', 'w', 'utf-8') as fd:
        writer = csv.DictWriter(fd, delimiter='\t', fieldnames=list(data[0].keys()))
        writer.writeheader()
        for r in data:
            writer.writerow(r)


if __name__ == '__main__':
    # global DATA_PATH
    DATA_PATH ='D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T2\\'
    load_price_detail('pricedetail.json')
    d1 = load_announce_data('AnnouncementsTrainSample.json')
    d2 = load_research_data('ResearchTrainSample.json')
    merge_data(d1, d2)
    gen_result('pred_result_1.txt', 'price_result.txt')

    DATA_PATH = 'D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\'
    load_price_detail('pricedetail.json')
    d1 = load_announce_data('AnnouncementsTrainSample.json')
    d2 = load_research_data('ResearchTrainSample.json')
    merge_data(d1, d2)
    gen_result('pred_result_1.txt', 'price_result.txt')