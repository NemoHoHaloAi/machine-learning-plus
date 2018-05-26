#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
project_name	My Decision Tree Classifier.
function	Build a decision tree for classification problem.
author		Ho Loong
date		2018-05-26
company		Aispeech,Inc.
ps              Please be pythonic.
'''

import sys
import os

class DecisionTree(object):
    def __init__(self):
        """
        初始化决策树
        """
        pass

    def fit(self, features_train, labels_train):
        """
        训练模型，此处的features_train类型为字典，key为特征名，value为各个样本对应该特征的值
        重点：根据信息增益来决策如何生成树结构，以及如何拆分样本
        """
        pass

    def predict(self, features_test):
        """
        预测数据，返回等长的结果列表
        """
        pass

    def score(self, features_test, labels_test):
        """
        评分，返回预测数据中正确的所占比例，0~1.0
        """
        pass

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

    clf = DecisionTree()
    clf.fit(features_train, labels_train)
    print clf.score(features_test, labels_test)

if __name__ == '__main__':
    main()
