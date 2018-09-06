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
    #xfmt = mdates.DateFormatter('%y-%m-%d')
    #ax.xaxis.set_major_formatter(xfmt)

    lines=[]
    with open('README.md', 'r') as fr:
        start = False
        end = False
        for line in fr.readlines():
            if '>>START<<' in line:
                start = True
            elif '>>END<<' in line:
                end = True
            if end:
                break
            if start:
                lines.append(line)

    print lines
    times = [line.split(',')[0] for line in lines]
    print times
    # 监督学习-76,监督学习-95
    dailies = []
    for line in lines:
        _from,_to = line.split(',')[1:]
        _from,_from_v = _from.split('-')
        _to,_to_v = _to.split('-')
        dailies.append(_to_v-_from_v if _to == _from else 100+_to_v-_from_v)
    print dailies
    plt.plot(times,dailies)

    plt.ylabel(u'学习量',fontproperties='SimHei')
    plt.xlabel(u'日期',fontproperties='SimHei')

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
