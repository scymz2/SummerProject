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
4. 现在因为RL做不出来，所以我想了另一个办法，将电池电量因素考虑进来，比如多少距离耗费多少电量，当
无人机到达一个新节点的时候就判断无人机的电量是否到达了预设危险值，是否还支持到下一个站点，或者需要立即返回来更换电池
需要的参数：
    电池百分比 battery level: 100%
    电池消耗效率：battery consume level: 0.1
    (可选)悬停电池消耗效率：
5.最后我们以时间来衡量效率
    影响时间的因素： 更换电池的时间，每个站点收集数据的停留时间，飞回起点以及起点飞下一个点的所需时间，其余飞行时间用路程/飞机速度计算
"""

import math
import random
import datetime
import copy
import numpy as np

EARTH_REDIUS = 6378137
pi = 3.1415926
VELOCITY = 10 # m/s
TIME_FOR_BATTERY_CHANGE = 300 # s
BATTERY_CONSUME = 0.067
START_POINT = [29.801243, 121.562457]
STATIONS = [[29.801135, 121.563133], [29.800235, 121.563446], [29.800769, 121.562406], [29.802068, 121.562487],
            [29.799767, 121.565542], [29.80015, 121.564544], [29.802247, 121.563245], [29.797815, 121.560359],
            [29.801755, 121.560063], [29.801953, 121.564465], [29.801579, 121.565629], [29.80148, 121.562626],
            [29.800737, 121.560633], [29.802348, 121.560778], [29.799546, 121.562976], [29.801447, 121.561482],
            [29.800359, 121.562595], [29.800619, 121.560055], [29.800615, 121.5635], [29.800958, 121.565426],
            [29.799665, 121.562034], [29.797673, 121.563997], [29.800899, 121.563818], [29.798036, 121.561168],
            [29.799277, 121.561778], [29.799673, 121.563257], [29.797894, 121.565685], [29.798208, 121.561159],
            [29.798694, 121.563739], [29.798647, 121.562654], [29.797938, 121.560869], [29.798085, 121.563757],
            [29.799304, 121.561088], [29.797868, 121.564714], [29.797863, 121.564812], [29.799832, 121.565616],
            [29.800552, 121.565618], [29.797562, 121.564239], [29.79799, 121.561588], [29.797982, 121.561666],
            [29.799544, 121.561793], [29.801015, 121.560319], [29.798758, 121.563236], [29.797851, 121.562985],
            [29.802266, 121.563291], [29.800882, 121.561796], [29.801141, 121.560711], [29.798323, 121.561627],
            [29.797461, 121.563352], [29.79738, 121.56476]]


class LocalClosestFirst:
    def __init__(self, start, stations):
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

        #电池变量
        self.Battery_level = 100
        self.Battery_low = 20
        self.Flight_time = 0


    def LCF(self):
        """
        lcf核心算法，从起始点开始找最近点,找到当前最近点就标记一下，然后找下一个，直到没有了就回到起点
        我们假设无人机的最低安全电量支持它从范围内的任意一点飞回出发点
        """
        path = []
        total_time = 0

        # print("test")
        # print(self.stations)

        for i in range(len(self.stations)):
            shortest, closest = self._lcf(self.cur)      # 找到
            segment_time = shortest / VELOCITY           # 计算片段时间
            predict_battery = self.Battery_level - (segment_time * BATTERY_CONSUME)  # 计算估计剩余消耗电量
            # 如果电量即将跌出安全范围就立即返回
            if(predict_battery <= self.Battery_low):
                # 记录返回以及前往下一个点的所需时间
                time_back = getDistance(self.start, self.cur) / VELOCITY
                time_next = getDistance(self.start, self.stations[closest]) / VELOCITY
                total_time += (time_back + time_next + TIME_FOR_BATTERY_CHANGE)
                #充电
                self.Battery_level = 100
            else:
                # 记录前往下一个点的所需时间
                total_time += segment_time
                self.Battery_level = predict_battery

            # 标记经过点
            self.visited[closest] = 1
            self.cur = self.stations[closest]

            path.append(closest)    # 将新找到的点加入path中

        # print(path)
        # 加入返回起始点的时间
        total_time += (getDistance(self.start, self.stations[closest]) / VELOCITY)
        return path, total_time

    def _lcf(self, cur):
        """
        这个函数找距离当前点最近的未被访问过的点
        """
        shortest = 999999999
        closest = -1
        for i in range(len(self.stations)):
            if(self.visited[i] == 0):
                if(getDistance(cur, self.stations[i]) < shortest):
                    closest = i
                    shortest = getDistance(cur, self.stations[i])

        return shortest, closest




class SimulatedAnnealing:

    # 构造方法
    def __init__(self, start, stations, lcf_path):
        assert len(start) > 1, "没有设定起始点"
        assert len(stations) > 1, "基站数量太少"

        self.start = start
        self.cur = start
        self.stations = stations
        self.lcf_path = lcf_path
        # 以_开头为私有变量,这边的参数主要为
        self._SA_TS = 200   # 起始温度
        self._SA_TF = 1     # 结束温度
        self._SA_BETA = 0.0000001
        self._SA_MAX_ITER = 200000000
        self._SA_MAX_TIME = 600
        self._SA_ITER_PER_T = 1
        # 电池变量
        self.Battery_level = 100
        self.Battery_low = 20
        self.Flight_time = 0


    # SA主函数
    def run_SA(self):
        iter = 0
        time_spend = 0
        temperature = self._SA_TS
        cur_path = self.lcf_path
        curr_time1 = datetime.datetime.now()

        while (iter < self._SA_MAX_ITER) & (time_spend < self._SA_MAX_TIME) & (temperature > self._SA_TF):
            # 随机取两个不同的station进行交换
            item = get2RandomInt(0, 49)
            new_path = self.swapNode(item[0], item[1], cur_path)
            """
            这边我们需要做的变化是
            1.将delta的值从路程编程时间
            """
            # 计算delta,即交换前后的差值
            delta = self.getTotalTime(cur_path) - self.getTotalTime(new_path)
            # 模拟退火，熵值概率交换
            if (delta > 0) | ((delta < 0) & (math.exp(delta/temperature) > random.uniform(0, 1))):
                cur_path = new_path
                if(iter % 100 ==0):
                    print("iter: " + str(iter))
                    print("delta: " + str(delta))
                    print("math.exp(delta/temperature): " + str(math.exp(delta / temperature)))
                    print("temperature: " + str(temperature) + " best obj: " + str(self.getTotalTime(cur_path)))

            temperature = temperature/(1 + self._SA_BETA*temperature)
            curr_time2 = datetime.datetime.now()
            time_spend = (curr_time2-curr_time1).seconds
            iter += 1

        return self.getTotalTime(cur_path)

    # 交换节点
    def swapNode(self, item1, item2, path):
        # 传进来的参数是异名同地址变量，所以需要copy一份
        new_path = copy.deepcopy(path)
        # 黑科技，一次赋值两个变量
        new_path[item1], new_path[item2] = new_path[item2], new_path[item1]
        return new_path


    # 这个函数用来计算整段路径需要花的总时间
    def getTotalTime(self, path):

        total_time = 0
        self.Battery_level = 100

        for i in range(len(self.stations)):
            next_index = path[i]
            segment_time = getDistance(self.cur, self.stations[next_index]) / VELOCITY
            predict_battery = self.Battery_level - (segment_time * BATTERY_CONSUME)
            #如果电量在到达下一个点之后跌出安全范围就返回充电一次
            if(predict_battery <= self.Battery_low):
                time_back = getDistance(self.start, self.cur) / VELOCITY
                time_next = getDistance(self.stations[next_index],self.start) / VELOCITY
                total_time += (time_back + time_next + TIME_FOR_BATTERY_CHANGE)
                self.Battery_level = 100 #充电
            else:
                #不然就飞到下一个点
                total_time += segment_time
                self.Battery_level = predict_battery
                self.cur = self.stations[next_index]
        #加入返回起始点的时间
        total_time += (getDistance(self.start,self.stations[next_index]) / VELOCITY)
        return total_time




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

# 获取当前路径总长度
def getTotalDistance(self, path):
    distance = 0
    # 计算起始点到第一个基站和最后一个基站返回起始点的距离
    start2first = getDistance(self.start, self.stations[path[0]])
    last2end = getDistance(self.stations[path[0]], self.start)

    for i in range(49):
        dis = getDistance(self.stations[path[i]], self.stations[path[i+1]])
        distance += dis

    distance += (start2first + last2end)
    return distance


if __name__ == "__main__":

    time = []

    lcf = LocalClosestFirst(START_POINT, STATIONS)
    lcf_path, total_time = lcf.LCF()
    print(total_time)

    #print(lcf_path)


    sa = SimulatedAnnealing(START_POINT, STATIONS, lcf_path)
    print(sa.getTotalTime(lcf_path))
    for i in range(1):
        time.append(sa.run_SA())
    print(time)



