#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:nicolasjam
@file:main.py
@time:2020/11/06
"""
import math

"""
任务简介
由于anylogic里面添加函数实在是太麻烦了，所以我人工将预设点坐标以及各种数据转出在这个程序里面计算
我需要什么
1.一个可以计算GPS坐标系距离的函数
2.一个SA算法
3.一个closest first算法
4.还需要一个RL训练模型（尽量找现成的）
5.我需要得出的结果，用这三种方法分别计算出的飞行距离，计算耗时
"""

import datetime
import time

EARTH_REDIUS = 6378.137
pi = 3.1415926

#散装函数先放在这边
curr_time1 = datetime.datetime.now()
curr_time2 = datetime.datetime.now()
print(curr_time2-curr_time1)

def GCJ2WGS(location):
 # location格式如下：locations[1] = "113.923745,22.530824"
     lon = float(location[0:location.find(",")])
     lat = float(location[location.find(",") + 1:len(location)])
     a = 6378245.0 # 克拉索夫斯基椭球参数长半轴a
     ee = 0.00669342162296594323 #克拉索夫斯基椭球参数第一偏心率平方
     PI = 3.14159265358979324 # 圆周率
     # 以下为转换公式
     x = lon - 105.0
     y = lat - 35.0
     # 经度
     dLon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x));
     dLon += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0;
     dLon += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0;
     dLon += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0;
     #纬度
     dLat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x));
     dLat += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0;
     dLat += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0;
     dLat += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0;
     radLat = lat / 180.0 * PI
     magic = math.sin(radLat)
     magic = 1 - ee * magic * magic
     sqrtMagic = math.sqrt(magic)
     dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI);
     dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI);
     wgsLon = lon - dLon
     wgsLat = lat - dLat
     return wgsLon,wgsLat



class LocalClosestFirst:
    #

class SimulatedAnnealing:

    #构造方法
    def __init__(self,lat,lng):
        assert lat.length >= 1 & lng.length >= 1, "坐标数量必须大于1"
        self.lat = lat
        self.lng = lng
        # 以_开头为私有变量,这边的参数主要为
        self._SA_TS = 500 #起始温度
        self._SA_TF = 1; #结束温度
        self._SA_BETA = 0.00000001
        self._SA_MAX_ITER = 200000000
        self._SA_ITER_PER_T = 1
        self._time_spent = 0

    def


def rad(d):
    return d * pi / 180.0

def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s

