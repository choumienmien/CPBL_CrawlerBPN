# -*- coding:utf-8 -*-
import csv
import random
import re
import time
from collections import OrderedDict

import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import datatype
import xmltodict
import urllib3
import itertools
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# class GameDetailSpider:
#     def __init__(self):
#         self.url = "https://www.cpbl.com.tw/box/index?gameSno={}&year={}&kindCode=A"
#         self.f = open("Crawl_GameDetail.csv", "w", newline="", encoding="utf_8_sig")
#         self.writer = csv.writer(self.f, delimiter=",")
#         # self.f = open("cpbl.json", "w", newline="", encoding="utf_8_sig")
#         self.ses = HTMLSession()
#         self.keys_str_lst = []
#         self.rows = []
#
#     def get_keys(self, url, regex):
#         res = self.ses.post(url, verify=False)
#         for l in res.text.split('\n'):
#             if 'RequestVerificationToken' in l:
#                 pattern = re.compile(regex, re.S)
#                 keys = pattern.findall(l)
#                 tuple_keys = (keys[0][0], keys[0][1], keys[0][2])
#                 self.token = tuple_keys[0]
#                 self.year = tuple_keys[2]
#                 self.gamesno = tuple_keys[1]
#
#     def get_data(self):
#         data = {}
#         # 需加入'defendStation'
#         res = self.ses.post(datatype.root + '/box/getlive',
#                             params={'__RequestVerificationToken': self.token,
#                                     'GameSno': self.gamesno,
#                                     'KindCode': 'A',
#                                     'Year': self.year}
#                             )
#
#         data['game_detail'] = json.loads(res.json()['GameDetailJson'])
#         # return data
#
#         if self.keys_str_lst == []:
#             dict_keys = list(list(data.values())[0][0].keys())
#             self.keys_str_lst.append(dict_keys)  # [['FieldAbbe', 'VisitingTeamName', 'HomeTeamName'.....]]
#             self.headers = self.keys_str_lst[0]  # ['FieldAbbe', 'VisitingTeamName', 'HomeTeamName'.....]
#
#         else:
#             pass
#
#         # 當天2場不同賽事進行(有些年會重覆同天場次之資料)
#         if len(list(data.values())[0]) > 1:
#             # print(len(list(data.values())[0]))
#             for i in list(data.values())[0]:
#                 self.rows.append(i)
#
#         else:
#             dict_values = list(data.values())[0][0]
#             self.rows.append(dict_values)
#
#         # self.writer.writerow(list(data.values()))
#         # json.dump(list(data.values()),self.f)
#
#     def save_data(self):
#         # print(self.rows)
#         # exit()
#         print(len(self.rows))
#         new_dict = {}
#         for di in self.rows:
#             # 屬於game_detail內容
#             new_dict[di['GameSno']] = {}
#             for k in di.keys():
#                 new_dict[di['GameSno']][k] = di[k]
#
#         writer = csv.DictWriter(self.f, self.headers)
#         writer.writeheader()
#         for i in new_dict.values():
#             writer.writerow(i)
#         print(len(new_dict))
#         # print('------------------------------')
#
#     def crawl(self):
#         regex = '<form action="/box/live" id="MainForm" method="post">.*?value="(.*?)".*?value="(.*?)".*?<input id="Year".*?value="(.*?)".*?'
#         # years = [2019,2020,2021,2022]
#         years = [2019]
#         for year in years:
#             for sno in range(1, 5):
#                 page_url = self.url.format(sno, year)
#                 self.get_keys(page_url, regex)
#                 self.get_data()
#                 # 控制頻率
#                 time.sleep(random.randint(1, 3))
#         self.save_data()
#         self.f.close()


# class ScoreBoardSpider:
#     def __init__(self):
#         self.url = "https://www.cpbl.com.tw/box/index?gameSno={}&year={}&kindCode=A"
#         self.f = open("Crawl_ScoreBoard.csv", "w", newline="", encoding="utf_8_sig")
#         self.writer = csv.writer(self.f, delimiter=",")
#         # self.f = open("cpbl.json", "w", newline="", encoding="utf_8_sig")
#         self.ses = HTMLSession()
#         self.keys_str_lst = []
#         self.rows = []
#
#     def get_keys(self, url, regex):
#         res = self.ses.post(url, verify=False)
#         for l in res.text.split('\n'):
#             if 'RequestVerificationToken' in l:
#                 pattern = re.compile(regex, re.S)
#                 keys = pattern.findall(l)
#                 tuple_keys = (keys[0][0], keys[0][1], keys[0][2])
#                 self.token = tuple_keys[0]
#                 self.year = tuple_keys[2]
#                 self.gamesno = tuple_keys[1]
#
#     def get_data(self):
#         data = {}
#         # 需加入'defendStation'
#         res = self.ses.post(datatype.root + '/box/getlive',
#                             params={'__RequestVerificationToken': self.token,
#                                     'GameSno': self.gamesno,
#                                     'KindCode': 'A',
#                                     'Year': self.year}
#                             )
#         # data['game_detail'] = json.loads(res.json()['GameDetailJson'])
#         data['score_board'] = json.loads(res.json()['ScoreboardJson'])
#         # data['live_log'] = json.loads(res.json()['LiveLogJson'])  # 無須取得相關數據(每個選手每次打擊狀況)
#         # data['batting'] = json.loads(res.json()['BattingJson'])
#         # # data['pitching'] = json.loads(res.json()['PitchingJson'])
#         # # data['first_Sno'] = json.loads(res.json()['FirstSnoJson'])#無須取得(這是當局每個選手的個人資訊)
#         # data['curt_game_detail'] = json.loads(res.json()['CurtGameDetailJson'])
#         # return data
#         # print(data)
#         # exit()
#
#         if self.keys_str_lst == []:
#             dict_keys = list(list(data.values())[0][0].keys())
#             self.keys_str_lst.append(dict_keys)  # [['FieldAbbe', 'VisitingTeamName', 'HomeTeamName'.....]]
#             self.headers = self.keys_str_lst[
#                 0]  # ['TeamName', 'TeamAbbr', 'TeamNo', 'VisitingHomeType', 'InningSeq'....]
#
#         else:
#             pass
#
#         # print(list(data.values())[0]) #[{'TeamName': '大高熊育樂股份有限公司', .....}] 當天比賽所有局數之數據
#         # print(list(data.values())[0][0]) #其中一局(EX:9局上)
#
#         # 當天有9*2局(但會有延長賽)
#         for i in list(data.values())[0][::-1]:
#             # print(self.year,self.gamesno,len(list(data.values())[0]))
#             self.rows.append(i)
#
#         # self.writer.writerow(list(data.values()))
#         # json.dump(list(data.values()),self.f)
#
#     def save_data(self):
#         print(len(self.rows))
#         new_dict = {}
#         for di in self.rows:
#             new_dict[di['CreateTime']] = {}
#             for k in di.keys():
#                 new_dict[di['CreateTime']][k] = di[k]
#
#         writer = csv.DictWriter(self.f, self.headers)
#         writer.writeheader()
#         for i in new_dict.values():
#             writer.writerow(i)
#         print(len(new_dict))
#         # print('------------------------------')
#
#     def crawl(self):
#         regex = '<form action="/box/live" id="MainForm" method="post">.*?value="(.*?)".*?value="(.*?)".*?<input id="Year".*?value="(.*?)".*?'
#         # years = [2019,2020,2021,2022]
#         years = [2019]
#         for year in years:
#             for sno in range(1, 229):
#                 page_url = self.url.format(sno, year)
#                 self.get_keys(page_url, regex)
#                 self.get_data()
#                 # 控制頻率
#                 time.sleep(random.randint(1, 3))
#         self.save_data()
#         self.f.close()


class CurtGameDetailSpider:
    def __init__(self):
        self.url = "https://www.cpbl.com.tw/box/index?gameSno={}&year={}&kindCode=A"
        # self.f = open("Crawl_CurtGameDetail.csv", "w", newline="", encoding="utf_8_sig")
        # self.writer = csv.writer(self.f, delimiter=",")
        # self.f = open("cpbl.json", "w", newline="", encoding="utf_8_sig")
        self.ses = HTMLSession()
        self.keys_str_lst = []
        self.rows = []

    def get_keys(self, url, regex):
        res = self.ses.post(url, verify=False)
        for l in res.text.split('\n'):
            if 'RequestVerificationToken' in l:
                pattern = re.compile(regex, re.S)
                keys = pattern.findall(l)
                tuple_keys = (keys[0][0], keys[0][1], keys[0][2])
                self.token = tuple_keys[0]
                self.year = tuple_keys[2]
                self.gamesno = tuple_keys[1]

    def get_data(self):
        data = {}
        # 需加入'defendStation'
        res = self.ses.post(datatype.root + '/box/getlive',
                            params={'__RequestVerificationToken': self.token,
                                    'GameSno': self.gamesno,
                                    'KindCode': 'A',
                                    'Year': self.year}
                            )


        data['curt_game_detail'] = json.loads(res.json()['CurtGameDetailJson'])
        # return data
        # print(data)
        # exit()

        if self.keys_str_lst == []:
            dict_keys = list(list(data.values())[0].keys())
            self.keys_str_lst.append(dict_keys)  # [['ErrorCnt', 'TotalHittingCnt', 'TotalHitCnt'.....]]
            self.headers = self.keys_str_lst[0]  # ['ErrorCnt', 'TotalHittingCnt', 'TotalHitCnt'....]
        else:
            pass
        # print(self.headers)
        # exit()

        # print(list(data.values())[0])
        # print('=======================================')
        # exit()


        for i in list(data.values()):
            self.rows.append(i)
            # print(self.rows)
            # exit()


    def save_data(self):
        # print(self.rows)
        # exit()
        # print(len(self.rows))
        new_dict = {}
        for di in self.rows:
            new_dict[di['Pkno']] = {}
            for k in di.keys():
                new_dict[di['Pkno']][k] = di[k]

        filename='Crawl_CurtGameDetail_{}.csv'
        self.f = open(filename.format(self.year), "w", newline="", encoding="utf_8_sig")
        # self.writer = csv.writer(self.f, delimiter=",")
        writer = csv.DictWriter(self.f, self.headers)
        writer.writeheader()
        for i in new_dict.values():
            writer.writerow(i)
        # print(len(new_dict))
        # print('------------------------------')

    def crawl(self):
        regex = '<form action="/box/live" id="MainForm" method="post">.*?value="(.*?)".*?value="(.*?)".*?<input id="Year".*?value="(.*?)".*?'
        self.years = [2022]
        # self.years = [2015,2016,2017,2018,2019,2020]
        for year in self.years:
            # self.rows = []
            for sno in range(1, 151):
                page_url = self.url.format(sno, year)
                self.get_keys(page_url, regex)
                self.get_data()
                # 控制頻率
                time.sleep(random.randint(1, 3))

        self.save_data()
        self.f.close()


class BattingSpider:
    def __init__(self):
        self.url = "https://www.cpbl.com.tw/box/index?gameSno={}&year={}&kindCode=A"
        # self.f = open("Crawl_Batting.csv", "w", newline="", encoding="utf_8_sig")
        # self.writer = csv.writer(self.f, delimiter=",")
        # self.f = open("cpbl.json", "w", newline="", encoding="utf_8_sig")
        self.ses = HTMLSession()
        self.keys_str_lst = []
        # self.rows = []

    def get_keys(self, url, regex):
        res = self.ses.post(url, verify=False)
        for l in res.text.split('\n'):
            if 'RequestVerificationToken' in l:
                pattern = re.compile(regex, re.S)
                keys = pattern.findall(l)
                tuple_keys = (keys[0][0], keys[0][1], keys[0][2])
                self.token = tuple_keys[0]
                self.year = tuple_keys[2]
                self.gamesno = tuple_keys[1]

    def get_data(self):
        data = {}
        # 需加入'defendStation'
        res = self.ses.post(datatype.root + '/box/getlive',
                            params={'__RequestVerificationToken': self.token,
                                    'GameSno': self.gamesno,
                                    'KindCode': 'A',
                                    'Year': self.year}
                            )

        data['batting'] = json.loads(res.json()['BattingJson'])
        # # data['pitching'] = json.loads(res.json()['PitchingJson'])
        # # data['first_Sno'] = json.loads(res.json()['FirstSnoJson'])
        # data['curt_game_detail'] = json.loads(res.json()['CurtGameDetailJson'])
        # return data
        # print(data)
        # exit()

        if self.keys_str_lst == []:
            dict_keys = list(list(data.values())[0][0].keys())
            self.keys_str_lst.append(dict_keys)  # [['ErrorCnt', 'TotalHittingCnt', 'TotalHitCnt'.....]]
            self.headers = self.keys_str_lst[0]  # ['ErrorCnt', 'TotalHittingCnt', 'TotalHitCnt'....]


        else:
            pass
        # print(self.headers)
        # exit()

        # print(list(data.values())[0]) #[{'ErrorCnt': 0, 'TotalHittingCnt': 2, 'TotalHitCnt': 2,  .....}] 當天比賽所有局數之數據
        # print('=======================================')
        # print(list(data.values())[0][0]) #2球隊各打擊手(EX:陳傑憲)
        # exit()

        # 2隊打擊球員
        for i in list(data.values())[0]:
            self.rows.append(i)

    def save_data(self):
        # print(len(self.rows))
        new_dict = {}
        for di in self.rows:
            new_dict[di['Pkno']] = {}
            for k in di.keys():
                new_dict[di['Pkno']][k] = di[k]

        filename='Crawl_Batting_{}.csv'
        self.f = open(filename.format(self.year), "w", newline="", encoding="utf_8_sig")
        # self.writer = csv.writer(self.f, delimiter=",")
        writer = csv.DictWriter(self.f, self.headers)
        writer.writeheader()
        for i in new_dict.values():
            writer.writerow(i)
        # print(len(new_dict))
        # print('------------------------------')

    def crawl(self):
        regex = '<form action="/box/live" id="MainForm" method="post">.*?value="(.*?)".*?value="(.*?)".*?<input id="Year".*?value="(.*?)".*?'
        years = [2022]
        # years = [2015,2016,2017,2018,2019,2020]
        for year in years:
            self.rows = []
            for sno in range(1, 151):
                page_url = self.url.format(sno, year)
                self.get_keys(page_url, regex)
                self.get_data()
                # 控制頻率
                time.sleep(random.randint(1, 3))
            self.save_data()
            self.f.close()


class PitchingSpider:
    def __init__(self):
        self.url = "https://www.cpbl.com.tw/box/index?gameSno={}&year={}&kindCode=A"
        # self.f = open("Crawl_Pitching.csv", "w", newline="", encoding="utf_8_sig")
        # self.writer = csv.writer(self.f, delimiter=",")
        # self.f = open("cpbl.json", "w", newline="", encoding="utf_8_sig")
        self.ses = HTMLSession()
        self.keys_str_lst = []
        # self.rows = []

    def get_keys(self, url, regex):
        res = self.ses.post(url, verify=False)
        for l in res.text.split('\n'):
            if 'RequestVerificationToken' in l:
                pattern = re.compile(regex, re.S)
                keys = pattern.findall(l)
                tuple_keys = (keys[0][0], keys[0][1], keys[0][2])
                self.token = tuple_keys[0]
                self.year = tuple_keys[2]
                self.gamesno = tuple_keys[1]

    def get_data(self):
        data = {}
        # 需加入'defendStation'
        res = self.ses.post(datatype.root + '/box/getlive',
                            params={'__RequestVerificationToken': self.token,
                                    'GameSno': self.gamesno,
                                    'KindCode': 'A',
                                    'Year': self.year}
                            )

        data['pitching'] = json.loads(res.json()['PitchingJson'])
        # # data['first_Sno'] = json.loads(res.json()['FirstSnoJson'])
        # data['curt_game_detail'] = json.loads(res.json()['CurtGameDetailJson'])
        # return data
        # print(data)
        # exit()

        if self.keys_str_lst == []:
            dict_keys = list(list(data.values())[0][0].keys())
            self.keys_str_lst.append(dict_keys)  # [['ErrorCnt', 'TotalHittingCnt', 'TotalHitCnt'.....]]
            self.headers = self.keys_str_lst[0]  # ['ErrorCnt', 'TotalHittingCnt', 'TotalHitCnt'....]


        else:
            pass
        # print(self.headers)
        # exit()

        # print(list(data.values())[0]) #[{'ErrorCnt': 0, 'TotalHittingCnt': 2, 'TotalHitCnt': 2,  .....}] 當天比賽所有局數之數據
        # print('=======================================')
        # print(list(data.values())[0][0]) #2球隊各打擊手(EX:陳傑憲)
        # exit()

        # 2隊打擊球員
        for i in list(data.values())[0]:
            self.rows.append(i)

    def save_data(self):
        # print(self.rows)
        # exit()
        # print(len(self.rows))
        new_dict = {}
        for di in self.rows:
            new_dict[di['Pkno']] = {}
            for k in di.keys():
                new_dict[di['Pkno']][k] = di[k]

        filename='Crawl_Pitching_{}.csv'
        self.f = open(filename.format(self.year), "w", newline="", encoding="utf_8_sig")
        # self.writer = csv.writer(self.f, delimiter=",")
        writer = csv.DictWriter(self.f, self.headers)
        writer.writeheader()
        for i in new_dict.values():
            writer.writerow(i)
        # print(len(new_dict))
        # print('------------------------------')

    def crawl(self):
        regex = '<form action="/box/live" id="MainForm" method="post">.*?value="(.*?)".*?value="(.*?)".*?<input id="Year".*?value="(.*?)".*?'
        years = [2022]
        #years = [2018,2019,2020]
        for year in years:
            self.rows = []
            for sno in range(150, 190):
                page_url = self.url.format(sno, year)
                self.get_keys(page_url, regex)
                self.get_data()
                # 控制頻率
                time.sleep(random.randint(1, 3))
            self.save_data()
            self.f.close()




if __name__ == '__main__':
    # spider = CurtGameDetailSpider()
    # spider = BattingSpider()
    spider = PitchingSpider()
    spider.crawl()
