#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:nicolasjam
@file:main.py
@time:2020/11/06

任务简介
由于anylogic里面添加函数实在是太麻烦了，所以我人工将预设点坐标以及各种数据转出在这个程序里面计算
我需要什么
1.一个可以计算GPS坐标系距离的函数
2.一个SA算法
3.一个closest first算法
4.还需要一个RL训练模型（尽量找现成的）
5.我需要得出的结果，用这三种方法分别计算出的飞行距离，计算耗时
"""

import math
import random
import datetime
import copy
import numpy as np

EARTH_REDIUS = 6378137
pi = 3.1415926
START_POINT = [29.801243, 121.562457]
STATIONS = [[29.801135, 121.563133], [29.800235, 121.563446], [29.800769, 121.562406], [29.802068, 121.562487], [29.799381, 121.565542], [29.80015, 121.564544], [29.802247, 121.563245], [29.797815, 121.560359], [29.801755, 121.560063], [29.801953, 121.564465], [29.801579, 121.565629], [29.80148, 121.562626], [29.800737, 121.560633], [29.802348, 121.560778], [29.799546, 121.562976], [29.801447, 121.561482], [29.800359, 121.562595], [29.800619, 121.560055], [29.800615, 121.5635], [29.800958, 121.565426], [29.799665, 121.562034], [29.797673, 121.563997], [29.800899, 121.563818], [29.798036, 121.561168], [29.799277, 121.561778], [29.799673, 121.563257], [29.797894, 121.565685], [29.798208, 121.561159], [29.798694, 121.563739], [29.798647, 121.562654], [29.797938, 121.560869], [29.798085, 121.563757], [29.799304, 121.561088], [29.797868, 121.564714], [29.797863, 121.564812], [29.799832, 121.565616], [29.800552, 121.565618], [29.797562, 121.564239], [29.79799, 121.561588], [29.797982, 121.561666], [29.799544, 121.561793], [29.801015, 121.560319], [29.798758, 121.563236], [29.797851, 121.562985], [29.802266, 121.563291], [29.800882, 121.561796], [29.801141, 121.560711], [29.798323, 121.561627], [29.797461, 121.563352], [29.79738, 121.56476]]


class LocalClosestFirst:
    def __init__(self,start,stations):
        """
        :param start: 起始点
        :param stations: 基站坐标
        """
        assert len(start) > 1, "没有设定起始点"
        assert len(stations) > 1, "基站数量太少"
        self.start = start
        self.stations = stations
        self.cur = start
        self.visited = np.zeros((len(stations),), dtype=int)


    def LCF(self):
        """
        lcf核心算法，从起始点开始找最近点,找到当前最近点就标记一下，然后找下一个，直到没有了就回到起点
        """
        path = []
        total_distance = 0

        #print("test")
        #print(self.stations)

        for i in range(len(self.stations)):
            shortest, closest = self._lcf(self.cur)     #找到
            total_distance += shortest      #计算总长
            path.append(closest)    #将新找到的点加入path中

        #print(path)
        return path

    def _lcf(self, cur):
        """
        这个函数找距离当前点最近的未被访问过的点
        """
        shortest = 999999999
        closest = -1
        for i in range(len(self.stations)):
            if(self.visited[i] == 0):

                if(getDistance(cur,self.stations[i]) < shortest):
                    closest = i
                    shortest = getDistance(cur, self.stations[i])

        self.visited[closest] = 1
        self.cur = self.stations[closest]
        return shortest, closest

        # 获取当前路径总长度
    def getTotalDistance(self, path):
        distance = 0
        # 计算起始点到第一个基站和最后一个基站返回起始点的距离
        start2first = getDistance(self.start, self.stations[path[0]])
        last2end = getDistance(self.stations[path[0]], self.start)

        for i in range(49):
            dis = getDistance(self.stations[path[i]], self.stations[path[i + 1]])
            distance += dis

        distance += (start2first + last2end)
        return distance


class SimulatedAnnealing:

    #构造方法
    def __init__(self,start,stations,lcf_path):
        assert len(start) > 1, "没有设定起始点"
        assert len(stations) > 1, "基站数量太少"

        self.start = start
        self.stations = stations
        self.lcf_path = lcf_path
        # 以_开头为私有变量,这边的参数主要为
        self._SA_TS = 200 #起始温度
        self._SA_TF = 7  #结束温度
        self._SA_BETA = 0.0000001
        self._SA_MAX_ITER = 200000000
        self._SA_MAX_TIME = 600
        self._SA_ITER_PER_T = 1


    #SA主函数
    def run_SA(self):
        iter = 0
        time_spend = 0
        temperature = self._SA_TS
        cur_path = self.lcf_path
        curr_time1 = datetime.datetime.now()

        while (iter < self._SA_MAX_ITER) & (time_spend < self._SA_MAX_TIME) & (temperature > self._SA_TF):
            #随机取两个不同的station进行交换
            item = get2RandomInt(0, 49)
            new_path = self.swapNode(item[0], item[1], cur_path)
            #计算delta,即交换前后的差值
            delta = self.getTotalDistance(cur_path) - self.getTotalDistance(new_path)
            #模拟退火，熵值概率交换
            if (delta > 0) | ((delta < 0) & (math.exp(delta/temperature) > random.uniform(0, 1))):
                cur_path = new_path
                if(iter%100 ==0):
                    print("iter: " + str(iter))
                    print("delta: " + str(delta))
                    print("math.exp(delta/temperature): " + str(math.exp(delta / temperature)))
                    print("temperature: " + str(temperature) + "best obj: " + str(self.getTotalDistance(cur_path)))

            temperature = temperature/(1+self._SA_BETA*temperature)
            curr_time2 = datetime.datetime.now()
            time_spend = (curr_time2-curr_time1).seconds
            iter += 1

    #交换节点
    def swapNode(self, item1, item2, path):
        #传进来的参数是异名同地址变量，所以需要copy一份
        new_path = copy.deepcopy(path)
        # 黑科技，一次赋值两个变量
        new_path[item1], new_path[item2] = new_path[item2], new_path[item1]
        return new_path



    #获取当前路径总长度
    def getTotalDistance(self, path):
        distance = 0
        #计算起始点到第一个基站和最后一个基站返回起始点的距离
        start2first = getDistance(self.start, self.stations[path[0]])
        last2end = getDistance(self.stations[path[0]], self.start)

        for i in range(49):
            dis = getDistance(self.stations[path[i]], self.stations[path[i+1]])
            distance += dis

        distance += (start2first + last2end)
        return distance

"""
这些是散装函数，大家共享
rad()
getDistance()
get2RandomInt()
"""

def rad(d):
    return d * pi / 180.0

def getDistance(position1, position2):
    lat1 = position1[0]
    lng1 = position1[1]
    lat2 = position2[0]
    lng2 = position2[1]
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(
    math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s

def get2RandomInt(bottom,top):
    s = []
    while (len(s) < 2):
        x = random.randint(bottom, top)
        if x not in s:
            s.append(x)
    return s


if __name__ == "__main__":

    lcf = LocalClosestFirst(START_POINT, STATIONS)
    lcf_path = lcf.LCF()
    #print(lcf_path)
    sa = SimulatedAnnealing(START_POINT, STATIONS, lcf_path)
    sa.run_SA()
    print("lcf obj:" + str(lcf.getTotalDistance(lcf_path)))



