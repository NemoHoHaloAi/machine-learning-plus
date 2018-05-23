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

from sklearn import svm

def enviroment_init():
    """
    程序运行环境初始化
    """
    pass

def test_svm():
    X = [[0,0],[1,1],[2,2]]
    Y = [0,1,0]

    clf = svm.SVC()
    clf.fit(X,Y)
    print clf.predict([[3,3]])

def main():
    """
    程序入口
    """
    enviroment_init()

    test_svm()

if __name__ == '__main__':
    main()
