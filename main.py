#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:nicolasjam
@file:main.py
@time:2020/11/06

任务简介
现在主要的代码基本都已经搞定了，现在我们要导出一系列数据来证明我们的paper
1.写出pure lcf and pure sa,即完全不考虑电量的飞行任务
2.添加一个其他指标及飞行过程钟消耗的总电量，这个可以更具飞行的距离来算
3.设计三个场景，基站的数量分别是16， 64 ，256
4.固定所有基站的位置，改变初始点的位置并研究出发点对于结果的影响
5.每个场景的SA测试100遍，并对几种算法进行比较
"""
import math
import random
import datetime
import copy

import numpy as np
import matplotlib.pyplot as plt
#地图范围
NUM = [16, 64, 256]
RANGE_LAT1 = 29.796666
RANGE_LAT2 = 29.803030
RANGE_LNG1 = 121.559098
RANGE_LNG2 = 121.566029

EARTH_REDIUS = 6378137
pi = 3.1415926
VELOCITY = 10  # m/s
# TIME_FOR_BATTERY_CHANGE = 300 # s
BATTERY_CONSUME = 0.67  # 0.067是标准25分钟电量，但是由于学校范围太小，单次任务基本不会耗干净电量，所以我们这里把耗速度加速10倍
START_POINT = [[29.801243, 121.562457], [29.800321, 121.566534], [29.798571, 121.559967], [29.800870, 121.564141], [29.803430, 121.561791],[29.801010, 121.560676],[29.798677, 121.563201],[29.796219, 121.563029],[29.799254, 121.565529],[29.799292, 121.563437]]
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

STATIONS_1 = [[29.801217, 121.562902], [29.800134, 121.563276], [29.800776, 121.562034], [29.802341, 121.562131], [29.799106, 121.565777], [29.800032, 121.564585], [29.802556, 121.563035], [29.79722, 121.55959], [29.801965, 121.559238], [29.802203, 121.564491], [29.801752, 121.565881], [29.801633, 121.562297], [29.800738, 121.559918], [29.802678, 121.560092], [29.799305, 121.562715], [29.801593, 121.560932]]
STATIONS_2 = [[29.801217, 121.562902], [29.800134, 121.563276], [29.800776, 121.562034], [29.802341, 121.562131], [29.799106, 121.565777], [29.800032, 121.564585], [29.802556, 121.563035], [29.79722, 121.55959], [29.801965, 121.559238], [29.802203, 121.564491], [29.801752, 121.565881], [29.801633, 121.562297], [29.800738, 121.559918], [29.802678, 121.560092], [29.799305, 121.562715], [29.801593, 121.560932], [29.800284, 121.56226], [29.800597, 121.559228], [29.800592, 121.56334], [29.801005, 121.565639], [29.799447, 121.56159], [29.797049, 121.563933], [29.800934, 121.563719], [29.797486, 121.560556], [29.798981, 121.561284], [29.799457, 121.56305], [29.797315, 121.565948], [29.797693, 121.560546], [29.798278, 121.563625], [29.798222, 121.56233], [29.797368, 121.5602], [29.797545, 121.563647], [29.799013, 121.560461], [29.797284, 121.564788], [29.797278, 121.564906], [29.799648, 121.565866], [29.800515, 121.565868], [29.796915, 121.564222], [29.797431, 121.561058], [29.797422, 121.561151], [29.799302, 121.561302], [29.801073, 121.559543], [29.798355, 121.563025], [29.797264, 121.562725], [29.80258, 121.56309], [29.800913, 121.561306], [29.801225, 121.560011], [29.797832, 121.561104], [29.796794, 121.563163], [29.796696, 121.564843], [29.798384, 121.563796], [29.802789, 121.564194], [29.800333, 121.560822], [29.800308, 121.563201], [29.802729, 121.560644], [29.802053, 121.562197], [29.798559, 121.563946], [29.799189, 121.564738], [29.800365, 121.565205], [29.801073, 121.565209], [29.799856, 121.564125], [29.800764, 121.565725], [29.800525, 121.562036], [29.798585, 121.559231]]
STATIONS_3 = [[29.801217, 121.562902], [29.800134, 121.563276], [29.800776, 121.562034], [29.802341, 121.562131], [29.799106, 121.565777], [29.800032, 121.564585], [29.802556, 121.563035], [29.79722, 121.55959], [29.801965, 121.559238], [29.802203, 121.564491], [29.801752, 121.565881], [29.801633, 121.562297], [29.800738, 121.559918], [29.802678, 121.560092], [29.799305, 121.562715], [29.801593, 121.560932], [29.800284, 121.56226], [29.800597, 121.559228], [29.800592, 121.56334], [29.801005, 121.565639], [29.799447, 121.56159], [29.797049, 121.563933], [29.800934, 121.563719], [29.797486, 121.560556], [29.798981, 121.561284], [29.799457, 121.56305], [29.797315, 121.565948], [29.797693, 121.560546], [29.798278, 121.563625], [29.798222, 121.56233], [29.797368, 121.5602], [29.797545, 121.563647], [29.799013, 121.560461], [29.797284, 121.564788], [29.797278, 121.564906], [29.799648, 121.565866], [29.800515, 121.565868], [29.796915, 121.564222], [29.797431, 121.561058], [29.797422, 121.561151], [29.799302, 121.561302], [29.801073, 121.559543], [29.798355, 121.563025], [29.797264, 121.562725], [29.80258, 121.56309], [29.800913, 121.561306], [29.801225, 121.560011], [29.797832, 121.561104], [29.796794, 121.563163], [29.796696, 121.564843], [29.798384, 121.563796], [29.802789, 121.564194], [29.800333, 121.560822], [29.800308, 121.563201], [29.802729, 121.560644], [29.802053, 121.562197], [29.798559, 121.563946], [29.799189, 121.564738], [29.800365, 121.565205], [29.801073, 121.565209], [29.799856, 121.564125], [29.800764, 121.565725], [29.800525, 121.562036], [29.798585, 121.559231], [29.798512, 121.563674], [29.799395, 121.563381], [29.798564, 121.560037], [29.800426, 121.563048], [29.800823, 121.563079], [29.799412, 121.563618], [29.799005, 121.565312], [29.802342, 121.562119], [29.801146, 121.564686], [29.802518, 121.559793], [29.803023, 121.564048], [29.802191, 121.560134], [29.800583, 121.560224], [29.802063, 121.559956], [29.800288, 121.564694], [29.797106, 121.56192], [29.799552, 121.563932], [29.80218, 121.564103], [29.802112, 121.565859], [29.798957, 121.559179], [29.797758, 121.564158], [29.797012, 121.562709], [29.796784, 121.560484], [29.798091, 121.564599], [29.802572, 121.561492], [29.796869, 121.56398], [29.800621, 121.560239], [29.79818, 121.563099], [29.800573, 121.565573], [29.80042, 121.56281], [29.798651, 121.564158], [29.798001, 121.561858], [29.802676, 121.560389], [29.799787, 121.564224], [29.798285, 121.560674], [29.799431, 121.5595], [29.801098, 121.561259], [29.797809, 121.561716], [29.797094, 121.559269], [29.799553, 121.563807], [29.802372, 121.562817], [29.798046, 121.565962], [29.798342, 121.563694], [29.801492, 121.559241], [29.799106, 121.561316], [29.801955, 121.563176], [29.80222, 121.563457], [29.801745, 121.560994], [29.80273, 121.560385], [29.798037, 121.563863], [29.801317, 121.565664], [29.798024, 121.560858], [29.796829, 121.56269], [29.799369, 121.560536], [29.799616, 121.561691], [29.8004, 121.561022], [29.797414, 121.565085], [29.797506, 121.562684], [29.799187, 121.564067], [29.797832, 121.563017], [29.799772, 121.560102], [29.802651, 121.561563], [29.80143, 121.564402], [29.797197, 121.565362], [29.800386, 121.562925], [29.798525, 121.565765], [29.797304, 121.560767], [29.802582, 121.559212], [29.801663, 121.563741], [29.800398, 121.561051], [29.799757, 121.559541], [29.802244, 121.565873], [29.802785, 121.561442], [29.802707, 121.560704], [29.801752, 121.565623], [29.80223, 121.563468], [29.802069, 121.561129], [29.79675, 121.563381], [29.797609, 121.561505], [29.79971, 121.565903], [29.800736, 121.562545], [29.797537, 121.561653], [29.797874, 121.564796], [29.798094, 121.562642], [29.802153, 121.559776], [29.802781, 121.565841], [29.801592, 121.565381], [29.797182, 121.561407], [29.798144, 121.561921], [29.797006, 121.560016], [29.796739, 121.564127], [29.797601, 121.564439], [29.797236, 121.559649], [29.798228, 121.563756], [29.800213, 121.562013], [29.801293, 121.565062], [29.797503, 121.560972], [29.798585, 121.559482], [29.799569, 121.560915], [29.801093, 121.563834], [29.799084, 121.561063], [29.801684, 121.560354], [29.801102, 121.559492], [29.801613, 121.564495], [29.799045, 121.560896], [29.798402, 121.563171], [29.79792, 121.561668], [29.79695, 121.562285], [29.797156, 121.564641], [29.798619, 121.562694], [29.802772, 121.563101], [29.796891, 121.563572], [29.799912, 121.562081], [29.801002, 121.562814], [29.797486, 121.561022], [29.802753, 121.56182], [29.802419, 121.560395], [29.799574, 121.562867], [29.799585, 121.565211], [29.799205, 121.564117], [29.801057, 121.565364], [29.798752, 121.563947], [29.800714, 121.564343], [29.797688, 121.560762], [29.80277, 121.564618], [29.800427, 121.562273], [29.799576, 121.565043], [29.80033, 121.565695], [29.80245, 121.564787], [29.797681, 121.56475], [29.799202, 121.563457], [29.799365, 121.559533], [29.802069, 121.560891], [29.802769, 121.559329], [29.798936, 121.561561], [29.797845, 121.559211], [29.80258, 121.561879], [29.802682, 121.559788], [29.799556, 121.565124], [29.798147, 121.561362], [29.796876, 121.563357], [29.799395, 121.559206], [29.798269, 121.55957], [29.798277, 121.560631], [29.796743, 121.560006], [29.800602, 121.559898], [29.802969, 121.565851], [29.797703, 121.561933], [29.799786, 121.563525], [29.797082, 121.565956], [29.798501, 121.564527], [29.800882, 121.560771], [29.800904, 121.560803], [29.799365, 121.562683], [29.798493, 121.562943], [29.799306, 121.563995], [29.80194, 121.561597], [29.796959, 121.565509], [29.798884, 121.56071], [29.802938, 121.564747], [29.802425, 121.565814], [29.802979, 121.561153], [29.79734, 121.560827], [29.798151, 121.565689], [29.797037, 121.563879], [29.802277, 121.564163], [29.799078, 121.560986], [29.801431, 121.561692], [29.79776, 121.560746], [29.798604, 121.562212], [29.798179, 121.564914], [29.802665, 121.56258], [29.802185, 121.563492], [29.801444, 121.565615], [29.802826, 121.563947], [29.799541, 121.56599], [29.798529, 121.559589], [29.799323, 121.560154], [29.800511, 121.560008], [29.802364, 121.561751], [29.800146, 121.565806], [29.800435, 121.561003], [29.799254, 121.565313], [29.798395, 121.562924], [29.799223, 121.562255], [29.799885, 121.56082], [29.79904, 121.561249], [29.801443, 121.562737], [29.802547, 121.56141], [29.796976, 121.565075], [29.799505, 121.560856], [29.798884, 121.559823]]



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

        # path及总时间
        self.path = []
        self.total_time = 0

    def pure_LCF(self):
        """
        这个是完完全全的LCF,不需要考虑电量因素，但是需要记录总共消耗的电量和消耗的总时间
        """
        for i in range(len(self.stations)):
            shortest, closest = self._lcf(self.cur) # 就近原则
            self.total_time += shortest / VELOCITY  # 记录时间
            self.cur = self.stations[closest]
            self.visited[closest] = 1  # 标记

        return self.total_time


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
        self._SA_TF = 5  # 结束温度
        self._SA_BETA = 0.0000001
        self._SA_MAX_ITER = 200000000
        self._SA_MAX_TIME = 1000
        self._SA_ITER_PER_T = 1
        # 电池变量
        self.Battery_level = 100
        self.Battery_low = 20
        self.Flight_time = 0

    # pure SA
    def run_pure_SA(self):
        iter = 0
        time_spend = 0
        temperature = self._SA_TS
        cur_path = self.lcf_path
        best_time = 9999999
        curr_time1 = datetime.datetime.now()

        print("pure SA start!")

        while (iter < self._SA_MAX_ITER) & (time_spend < self._SA_MAX_TIME) & (temperature > self._SA_TF):
            # 随机取两个不同的station进行交换
            item = get2RandomInt(0, len(self.stations)-1)
            new_path = self.swapNode(item[0], item[1], cur_path)
            #计算path的总距离(不回初始点)
            cur_time = self.getPathDistance(cur_path) / VELOCITY
            new_time = self.getPathDistance(new_path) / VELOCITY
            # 计算delta,交换前后的差值
            delta = (cur_time - new_time) * 5  # 这边的delta有助于收敛
            # 模拟退火
            if (delta > 0) | ((delta < 0) & (math.exp(delta / temperature) > random.uniform(0, 1))):
                cur_path = new_path
            cur_time = self.getPathDistance(cur_path) / VELOCITY
            if iter % 10000 == 0:
                print("iter: " + str(iter))
                print("delta: " + str(delta))
                print("math.exp(delta/temperature): " + str(math.exp(delta / temperature)))
                print("temperature: " + str(temperature) + " best obj: " + str(best_time))
                print("cur dis: " + str(cur_time))

            # 记录历史最佳
            if cur_time < best_time:
                best_time = cur_time

            temperature = temperature / (1 + self._SA_BETA * temperature)
            curr_time2 = datetime.datetime.now()
            time_spend = (curr_time2 - curr_time1).seconds
            iter += 1

        return best_time

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
            item = get2RandomInt(0, len(self.stations)-1)
            new_path = self.swapNode(item[0], item[1], cur_path)
            # 计算delta,即交换前后的差值
            delta = (self.getTotalTime(cur_path) - self.getTotalTime(new_path)) * 5 # delta*10有助于收敛，不然太不稳定
            # 模拟退火，熵值概率交换
            if (delta > 0) | ((delta < 0) & (math.exp(delta / temperature) > random.uniform(0, 1))):
                cur_path = new_path
            cur_time = self.getTotalTime(cur_path)
            if iter % 10000 == 0:
                print("iter: " + str(iter))
                print("delta: " + str(delta))
                print("math.exp(delta/temperature): " + str(math.exp(delta / temperature)))
                print("temperature: " + str(temperature) + " best obj: " + str(best_time))
                print("cur dis: " + str(cur_time))

            # 记录历史最佳
            if cur_time < best_time:
                print(cur_time <= best_time)
                print(best_time)
                print(cur_time)
                best_time = cur_time

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

    def getPathDistance(self,path):
        """
        :param path:  path
        :return:  the pure distance of the path, excluding the distance for recharge
        """
        distance = 0
        # 先计算出发点到第一个点的距离
        distance += getDistance(self.start, self.stations[path[0]])
        for i in range(len(path) - 1):
            distance += getDistance(self.stations[path[i]],self.stations[path[i+1]])

        # 再计算回到出发点的距离
        distance += getDistance(self.stations[path[len(path)-1]], self.start)

        return distance


"""
这些是散装函数，大家共享
rad()
getDistance()
get2RandomInt()
"""
def generateNodes(num, rangeLat1, rangeLat2,rangeLng1,rangeLng2):
    random = np.random.RandomState(0)  # RandomState生成随机数种子，每次随机都是一样的，因为种子确定了
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


#打图
def plotGragh_A(x_data, y_data,x_label, y_label, title):
    # 设置绘图风格（不妨使用R语言中的ggplot2风格）
    plt.style.use('ggplot')
    # 绘制条形图
    plt.bar(x=range(len(y_data)),  # 指定条形图x轴的刻度值
            height=y_data,  # 指定条形图y轴的数值
            tick_label=x_data,  # 指定条形图x轴的刻度标签
            color='steelblue',  # 指定条形图的填充色
            width=0.6
            )

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)

    for x, y in enumerate(y_data):
        plt.text(x, y + 0.1, '%s' % round(y, 1), ha='center')

    plt.show()



if __name__ == "__main__":


    time = []
    p_lcf = LocalClosestFirst(START_POINT[0], STATIONS_2)
    lcf = LocalClosestFirst(START_POINT[0], STATIONS_2)
    lcf2 = LocalClosestFirst(START_POINT[0], STATIONS_2)
    lcf_path, lcf_time = lcf.LCF()
    lcf2_time = lcf2.LCF_2()
    time = p_lcf.pure_LCF()
    print("lcf_pure: " + str(time))
    print("lcf_bc:   " + str(lcf_time))
    print("lcf_bfn:  " + str(lcf2_time))

    #sa = SimulatedAnnealing(START_POINT[0], STATIONS_1, lcf_path)
    pure_sa = SimulatedAnnealing(START_POINT[0], STATIONS_2, lcf_path)
    print("pure_sa: " + str(pure_sa.run_pure_SA()))
    #print("sa:      " + str(sa.run_SA()))




    """
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
    
    lcf = LocalClosestFirst(START_POINT[0], STATIONS)
    lcf_path, lcf_time = lcf.LCF()
    pure_sa = SimulatedAnnealing(START_POINT[0], STATIONS, lcf_path)
    print(pure_sa.run_pure_SA())
    """

    """
    N = 9

    # 任务一:分别打出 16个点， 64个点， 256个点在三个不同起点的数值
    lcf_16 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[0], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))
    lcf_64 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[1], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))
    lcf_256 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[2], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))

    # pure lcf
    time_16 = lcf_16.pure_LCF()
    time_64 = lcf_64.pure_LCF()
    time_256 = lcf_256.pure_LCF()

    # 重新初始化，避免数据相互影响（这步可以通过重构代码来省略）
    lcf_16 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[0], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))
    lcf_64 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[1], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))
    lcf_256 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[2], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))

    # lcf_bc
    path_16, time_bc_16 = lcf_16.LCF()
    path_64, time_bc_64 = lcf_64.LCF()
    path_256, time_bc_256 = lcf_256.LCF()

    # 重新初始化，避免数据相互影响（这步可以通过重构代码来省略）
    lcf_16 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[0], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))
    lcf_64 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[1], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))
    lcf_256 = LocalClosestFirst(START_POINT[N], generateNodes(NUM[2], RANGE_LAT1, RANGE_LAT2, RANGE_LNG1, RANGE_LNG2))

    # lcf_bfn
    time_bfn_16 = lcf_16.LCF_2()
    time_bfn_64 = lcf_64.LCF_2()
    time_bfn_256 = lcf_256.LCF_2()


    # 打图所需数据
    number = ('16', '64', '256')
    height1 = [time_16, time_64, time_256]
    height2 = [time_bc_16, time_bc_64, time_bc_256]
    height3 = [time_bfn_16, time_bfn_64, time_bfn_256]

    # 打印第一张图
    fig = plt.figure(1)
    plotGragh_A(number, height1, x_label="number of stations", y_label="Task Time (seconds)", title="PURE LCF S" + str(N+1))
    fig.savefig('lcf_pure_s'+ str(N+1) + '.png', dpi=fig.dpi)
    # 打印第二张图
    fig = plt.figure(2)
    plotGragh_A(number, height2, x_label="number of stations", y_label="Task Time (seconds)", title="LCF BC S" + str(N+1))
    fig.savefig('lcf_bc_s' + str(N+1) + '.png', dpi=fig.dpi)
    # 打印第三张图
    fig = plt.figure(3)
    plotGragh_A(number, height3, x_label="number of stations", y_label="Task Time (seconds)", title="LCF BFN S" + str(N+1))
    fig.savefig('lcf_bfn_s' + str(N+1) + '.png', dpi=fig.dpi)

    """