#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:nicolasjam
@file:temp.py
@time:2020/11/06
"""


import numpy as np
random = np.random.RandomState(0)#RandomState生成随机数种子
positions = []
for i in range(50):#随机数个数
    a = round(random.uniform(121.559946, 121.565753),6)#随机数范围
    b = round(random.uniform(29.797355,29.802640),6)
    positions.append([a,b])

