#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:nicolasjam
@file:temp.py
@time:2020/11/06
"""


import numpy as np
random = np.random.RandomState(0)#RandomState生成随机数种子
for i in range(200):#随机数个数
    a = random.uniform(-0.1, 0.1)#随机数范围
    print round(a, 2)#随机数精度要求