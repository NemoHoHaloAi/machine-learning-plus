#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
project_name	xxx
function	xxx
author		Ho Loong
date		xxxx-xx-xx
company		Aispeech,Inc.
ps              Please be pythonic.
'''

import sys
import os

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

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()

    reg.fit([[0,1],[1,1],[2,2]],[0,1,1])

    print reg.predict([[1,2]])

if __name__ == '__main__':
    main()
