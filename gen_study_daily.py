#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
project_name	绘制学习时间积累图，横坐标为日期，纵坐标为累计的学习量
function	xxx
author		Ho Loong
date		2018-09-06
company		Aispeech,Inc.
ps              Please be pythonic.
'''

import sys
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def enviroment_init():
    """
    程序运行环境初始化
    """
    pass

def main():
    """
    程序入口
    """
    enviroment_init()

    # 数据预处理
    lines=[]
    with open('README.md', 'r') as fr:
        start = False
        end = False
        for line in fr.readlines():
            if '>>START<<' in line:
                start = True
                continue
            elif '>>END<<' in line:
                end = True
            if end:
                break
            if start:
                lines.append(line)

    # 解析时间字段
    times = [line.split(',')[0].strip() for line in lines]
    times = [datetime.strptime(d, '%Y/%m/%d').date() for d in times]
    print times

    # 解析进度数据
    dailies = []
    dailies_acc = []
    offset = 0
    for line in lines:
        _from,_to = line.split(',')[1:]
        _from,_from_v = _from.split('-')
        _to,_to_v = _to.split('-')
        if _from != _to:
            offset = offset + 100
        _from_v = float(_from_v)
        _to_v = float(_to_v)
        dailies.append(_to_v - _from_v if _to == _from else 100+_to_v-_from_v)
        dailies_acc.append(_to_v + offset)
    print dailies
    print dailies_acc


    plt.figure(12, figsize=(15,15))

    # 绘制折线图，展示学习的积累情况，主要观察学习速率
    plt.subplot(121)
    plt.plot(times,dailies_acc)
    ax = plt.gca()
    xfmt = mdates.DateFormatter('%y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylabel(u'study_daily',fontproperties='SimHei')
    plt.xlabel(u'date',fontproperties='SimHei')
    plt.ylim((0, 500))
    plt.grid(True)

    for label in ax.xaxis.get_ticklabels():
       label.set_rotation(30)

    # 绘制直方图，体现每日学习量
    plt.subplot(122)
    plt.bar(times,dailies)
    ax = plt.gca()
    xfmt = mdates.DateFormatter('%y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylabel(u'study_daily',fontproperties='SimHei')
    plt.xlabel(u'date',fontproperties='SimHei')
    plt.grid(True)

    for label in ax.xaxis.get_ticklabels():
       label.set_rotation(30)

    # 如果保存的话就不能show，否则会导致保存的是一片空白，应该可以解决吧，不过对我没影响，直接注释
    #plt.show()

    # 自动保存图片到当前目录下
    plt.savefig('./study_daily.png')

if __name__ == '__main__':
    main()
