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
import matplotlib.pyplot as plt

EARTH_REDIUS = 6378137
pi = 3.1415926
VELOCITY = 10  # m/s
# TIME_FOR_BATTERY_CHANGE = 300 # s
BATTERY_CONSUME = 0.67  # 0.067是标准25分钟电量，但是由于学校范围太小，单次任务基本不会耗干净电量，所以我们这里把耗速度加速10倍
START_POINT = [[29.801243, 121.562457], [29.800321, 121.566534], [29.798571, 121.559967]]
STATIONS = [[29.801379, 121.563115], [29.800436, 121.563456], [29.800996, 121.562324],
            [29.802357, 121.562412], [29.799543, 121.565736], [29.800348, 121.56465],
            [29.802544, 121.563236], [29.797902, 121.560096], [29.802029, 121.559775],
            [29.802236, 121.564564], [29.801844, 121.565831], [29.801741, 121.562563],
            [29.800963, 121.560394], [29.80265, 121.560553], [29.799716, 121.562945],
            [29.801706, 121.561319], [29.800567, 121.562529], [29.800839, 121.559766],
            [29.800835, 121.563515], [29.801195, 121.565611], [29.799839, 121.561919],
            [29.797753, 121.564055], [29.801133, 121.56386], [29.798134, 121.560976],
            [29.799434, 121.56164], [29.799848, 121.56325], [29.797985, 121.565893],
            [29.798313, 121.560967], [29.798822, 121.563774], [29.798773, 121.562594],
            [29.798031, 121.560652], [29.798185, 121.563794], [29.799461, 121.560889],
            [29.797958, 121.564835], [29.797952, 121.564942], [29.800014, 121.565817],
            [29.800768, 121.565819], [29.797637, 121.564318], [29.798085, 121.561434],
            [29.798077, 121.561518], [29.799713, 121.561656], [29.801254, 121.560052],
            [29.798889, 121.563227], [29.79794, 121.562953], [29.802565, 121.563286],
            [29.801115, 121.56166], [29.801386, 121.56048], [29.798434, 121.561476],
            [29.797531, 121.563353], [29.797446, 121.564885]]


class LocalClosestFirst:
    """
    Local closest first 分成两个算法，第一个是先计算出，如果电不够就返回然后向下一个点出发
    第二个算法是先判断电量够不够，如果不够就返航，然后再从起点找最近的点出发
    """

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

        # 电池变量
        self.Battery_level = 100
        self.Battery_low = 20
        self.Flight_time = 0

        # path及总时间
        self.path = []
        self.total_time = 0

    def LCF(self):
        """
        lcf核心算法，从起始点开始找最近点,找到当前最近点就标记一下，然后找下一个，直到没有了就回到起点
        我们假设无人机的最低安全电量支持它从范围内的任意一点飞回出发点
        """

        for i in range(len(self.stations)):
            shortest, closest = self._lcf(self.cur)  # 找到
            segment_time = shortest / VELOCITY  # 计算片段时间
            # print("segment_time: " + str(segment_time))
            predict_battery = self.Battery_level - (segment_time * BATTERY_CONSUME)  # 计算估计剩余消耗电量
            # 如果电量即将跌出安全范围就立即返回
            if predict_battery <= self.Battery_low:
                # 记录返回以及前往下一个点的所需时间
                time_back = getDistance(self.start, self.cur) / VELOCITY
                time_next = getDistance(self.start, self.stations[closest]) / VELOCITY
                self.total_time += (time_back + time_next)
                self.Battery_level = 100  # 充电
                # print("charge")
            else:
                # 记录前往下一个点的所需时间
                self.total_time += segment_time
                self.Battery_level = predict_battery

            # 标记经过点
            self.visited[closest] = 1
            self.cur = self.stations[closest]
            self.path.append(closest)  # 将新找到的点加入path中
            # print(str(closest) + ": " + str(self.total_time))

        # print(path)
        # 加入返回起始点的时间
        self.total_time += (getDistance(self.start, self.stations[closest]) / VELOCITY)
        return self.path, self.total_time

    def LCF_2(self):
        """
        LCF变体，没电回到起点之后直接找离起点最近没访问过的点
        逻辑：从出发点开始，先找到最近的点，计算到那个所需要花费的电量和时间，
        然后在这个过程中，先判断电量是否充足以保证到达下一个点，如果可以的话
        """
        for i in range(len(self.stations)):
            shortest, closest = self._lcf(self.cur)  # 找到离当前点最近的未被访问的点
            segment_time = shortest / VELOCITY  # 计算片段时间
            predict_battery = self.Battery_level - (segment_time * BATTERY_CONSUME)  # 计算估计剩余消耗电量
            # 如果电量即将跌出安全范围就立即返回
            if predict_battery <= self.Battery_low:
                time_back = getDistance(self.start, self.cur) / VELOCITY  # 回到起点的时间
                self.total_time += time_back  # 统计时间
                self.Battery_level = 100  # 充电
                #然后再找离出发点最近的未被访问的点
                shortest, closest = self._lcf(self.start)
                self.total_time += shortest / VELOCITY  # 计算片段时间
                self.Battery_level -= (segment_time * BATTERY_CONSUME)  # 计算剩余电量
            else:
                # 记录前往下一个点的所需时间
                self.total_time += segment_time
                self.Battery_level = predict_battery
                # 标记经过点
                self.visited[closest] = 1
                self.cur = self.stations[closest]
        #计算回到原点的时间
        self.total_time += (getDistance(self.start, self.stations[closest]) / VELOCITY)
        return self.total_time

    def _lcf(self, cur):
        """
        这个函数找距离当前点最近的未被访问过的点
        """
        shortest = 999999999
        closest = -1
        for i in range(len(self.stations)):
            if self.visited[i] == 0:
                if getDistance(cur, self.stations[i]) < shortest:
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
        self._SA_TS = 200  # 起始温度
        self._SA_TF = 1  # 结束温度
        self._SA_BETA = 0.0000001
        self._SA_MAX_ITER = 200000000
        self._SA_MAX_TIME = 1000
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
        best_time = 9999999
        curr_time1 = datetime.datetime.now()

        print("SA start!")

        while (iter < self._SA_MAX_ITER) & (time_spend < self._SA_MAX_TIME) & (temperature > self._SA_TF):
            # 随机取两个不同的station进行交换
            item = get2RandomInt(0, 49)
            new_path = self.swapNode(item[0], item[1], cur_path)
            # 计算delta,即交换前后的差值
            delta = (self.getTotalTime(cur_path) - self.getTotalTime(new_path)) * 10 #delta*10有助于收敛，不然太不稳定
            # 模拟退火，熵值概率交换
            if (delta > 0) | ((delta < 0) & (math.exp(delta / temperature) > random.uniform(0, 1))):
                cur_path = new_path
                if iter % 1000 == 0:
                    print("iter: " + str(iter))
                    print("delta: " + str(delta))
                    print("math.exp(delta/temperature): " + str(math.exp(delta / temperature)))
                    print("temperature: " + str(temperature) + " best obj: " + str(best_time))
                    print("cur dis: " + str(self.getTotalTime(cur_path)))

            # 记录历史最佳
            if self.getTotalTime(cur_path) <= best_time:
                best_time = self.getTotalTime(cur_path)

            temperature = temperature / (1 + self._SA_BETA * temperature)
            curr_time2 = datetime.datetime.now()
            time_spend = (curr_time2 - curr_time1).seconds
            iter += 1

        return best_time

    # 交换节点
    def swapNode(self, item1, item2, path):
        # 传进来的参数是异名同地址变量，所以需要copy一份
        new_path = copy.deepcopy(path)
        # 黑科技，一次赋值两个变量
        new_path[item1], new_path[item2] = new_path[item2], new_path[item1]
        return new_path

    # 这个函数用来计算整段路径需要花的总时间
    def getTotalTime(self, path):

        self.Flight_time = 0
        self.Battery_level = 100

        for i in range(len(self.stations)):
            next_index = path[i]
            segment_time = getDistance(self.cur, self.stations[next_index]) / VELOCITY
            predict_battery = self.Battery_level - (segment_time * BATTERY_CONSUME)
            # 如果电量在到达下一个点之后跌出安全范围就返回充电一次
            if predict_battery <= self.Battery_low:
                time_back = getDistance(self.start, self.cur) / VELOCITY
                time_next = getDistance(self.stations[next_index], self.start) / VELOCITY
                self.Flight_time += (time_back + time_next)
                self.Battery_level = 100  # 充电
            else:
                # 不然就飞到下一个点
                self.Flight_time += segment_time
                self.Battery_level = predict_battery

            self.cur = self.stations[next_index]
        # 加入返回起始点的时间
        self.Flight_time += (getDistance(self.start, self.stations[next_index]) / VELOCITY)
        return self.Flight_time


"""
这些是散装函数，大家共享
rad()
getDistance()
get2RandomInt()
"""
def generateNodes(num, rangeLat1, rangeLat2,rangeLng1,rangeLng2):
    random = np.random.RandomState(0)  # RandomState生成随机数种子
    positions = []
    for i in range(num):  # 随机数个数
        a = round(random.uniform(rangeLng1, rangeLng2), 6)  # 随机数范围
        b = round(random.uniform(rangeLat1, rangeLat2), 6)
        positions.append([b, a])

    return positions

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


def get2RandomInt(bottom, top):
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
        dis = getDistance(self.stations[path[i]], self.stations[path[i + 1]])
        distance += dis

    distance += (start2first + last2end)
    return distance


if __name__ == "__main__":

    time = []

    lcf = LocalClosestFirst(START_POINT[0], STATIONS)
    lcf2 = LocalClosestFirst(START_POINT[0], STATIONS)
    lcf_path, lcf_time = lcf.LCF()
    lcf2_time = lcf2.LCF_2()

    num_list = [lcf_time, lcf2_time]
    name_list = ['lcf', 'lcf2']
    plt.bar(range(len(num_list)), num_list, fc= 'b', tick_label=name_list)
    plt.show()

    print(lcf_time)
    print(lcf2_time)

    # print(lcf_path)

    sa = SimulatedAnnealing(START_POINT[0], STATIONS, lcf_path)
    print(sa.getTotalTime(lcf_path))

    for i in range(1):
        time.append(sa.run_SA())
    print(time)

