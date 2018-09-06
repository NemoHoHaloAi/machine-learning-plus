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

    times = [line.split(',')[0].strip() for line in lines]
    times = [datetime.strptime(d, '%Y/%m/%d').date() for d in times]
    print times
    # 监督学习-76,监督学习-95
    dailies = []
    offset = 0
    for line in lines:
        _from,_to = line.split(',')[1:]
        _from,_from_v = _from.split('-')
        _to,_to_v = _to.split('-')
        if _from != _to:
            offset = offset + 100
        _from_v = float(_from_v)
        _to_v = float(_to_v)
        dailies.append(_to_v + offset)
    print dailies
    plt.plot(times,dailies)

    ax = plt.gca()
    xfmt = mdates.DateFormatter('%y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)

    plt.ylabel(u'study_daily',fontproperties='SimHei')
    plt.xlabel(u'date',fontproperties='SimHei')

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
