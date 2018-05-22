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

import numpy as np
from sklearn.naive_bayes import GaussianNB

def enviroment_init():
    """
    程序运行环境初始化
    """
    pass

def test_gaussian_naive_bayes():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    clf = GaussianNB()
    clf.fit(X, Y)
    print(clf.predict([[-0.8, -1]]))

def main():
    """
    程序入口
    """
    enviroment_init()

    test_gaussian_naive_bayes()   

if __name__ == '__main__':
    main()
