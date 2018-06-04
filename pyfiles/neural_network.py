#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
project_name	Neural Network Implemented
function	Build a class for Two Layer Neural Network, to do something like train and predict and else.
author		Ho Loong
date		2018-06-04
company		Aispeech,Inc.
ps              Please be pythonic.
'''

import sys
import os

import numpy as np
from sklearn.datasets import *

def enviroment_init():
    """
    enviroment init.
    """
    pass

def sigmoid(y):
    """
    激活函数，此处使用sigmoid，该函数优势是利于求导，激活函数是神经元中的计算部分.

    Args:
        y -- sum of every input feature multi his weight.

    Formula:
        sigmoid -- 1/(1+e^(-y))
    """
    return 1. / (1+np.exp(-y))

def sigmoid_prime(y):
    """
    sigmoid函数计算导数.

    Args:
        y -- sum of every input feature multi his weight.

    Formula:
        sigmoid -- f(y)=1/(1+e^(-y)), sigmoid函数计算导数有:f(y)'=f(y)(1-f(y))
    """
    return sigmoid(y)*(1-sigmoid(y))

def loss(real, target):
    """
    测量错误的损失函数；

    Args:
        real -- 真实输出
        target -- 网络给出的结果输出

    Formula:
        误差平方和 -- 0.5*sum((Xi-Yi)^2)
    """
    return 0.5 * sum([(real[i] - target[i])**2 for i in range(len(real))])

class Network(object):
    def __init__(self, sizes):
        """
        构造函数，接受参数指定的各层神经元个数，以及构造随机的biases和weights；
        
        Args:
            sizes -- 表示每一层的神经元个数，例如[2,3,1]表示输入层有两个，中间层有3个，输出层有1个；
        """
        self.num_layers = len(sizes) # 网络层数
        self.sizes = sizes # 此处都是两层网络，因此此处表示的就是输入层、中间层、输出层神经元个数
        # 偏差数据集，生成中间层结果和输出层结果中都需要该向量，因此长度为2，每一个向量的长度生成结果的长度，假如生成中间层的bias，那么该bias长度就是中间层的神经元个数
        self.biases = [np.random.randn(row, 1) for row in sizes[1:]] 
        # 权重数据集，权重向量，例如对于生成中间层的权重向量来说，输入层长度为n，所以权重向量的列长为n，中间层输出为m，所以权重向量的行长为m
        self.weights = [np.random.randn(row, col) for row, col in zip(sizes[1:], sizes[:-1])] 

    def feedforward(self, x):
        """
        前向传播：数据从输入层到输出层，经过各种非线性变化的过程

        Args:
            x -- 输入特征向量
        """
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w,x)+b) # 向量点乘代替了所有输入特征与对应的权重相乘再求和的过程，这也是向量计算的优势，不用循环啊
        return x
        
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """随机梯度下降"""
        if test_data:
            print "{0}: {1} / {2}".format('梯度下降前', self.evaluate(test_data), len(test_data))
        n = len(training_data)
        for j in xrange(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), len(test_data))
            else:
                print "Epoch {0} complete".format(j)
        
    def _backprop(self, x, y):
	"""返回一个元组(nabla_b, nabla_w)代表目标函数的梯度."""
	nabla_b = [np.zeros(b.shape) for b in self.biases]
	nabla_w = [np.zeros(w.shape) for w in self.weights]
	# feedforward
	activation = x
	activations = [x] # list to store all the activations, layer by layer
	zs = [] # list to store all the z vectors, layer by layer
	for b, w in zip(self.biases, self.weights):
	    z = np.dot(w, activation)+b
	    zs.append(z)
	    activation = sigmoid(z)
	    activations.append(activation)
	# backward pass
	delta = self._cost_derivative(activations[-1], y) * \
	    sigmoid_prime(zs[-1])
	nabla_b[-1] = delta
	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
	"""l = 1 表示最后一层神经元，l = 2 是倒数第二层神经元, 依此类推."""
	for l in xrange(2, self.num_layers):
	    z = zs[-l]
	    sp = sigmoid_prime(z)
	    delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
	    nabla_b[-l] = delta
	    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	return (nabla_b, nabla_w)

    def _update_mini_batch(self, mini_batch, eta):
        """使用后向传播算法进行参数更新.mini_batch是一个元组(x, y)的列表、eta是学习速率"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
                       
    def evaluate(self, test_data):
        """返回分类正确的个数"""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def _cost_derivative(self, output_activations, y):
        return (output_activations-y)

def main():
    """
    程序入口
    """
    enviroment_init()
    
    def vectorized_result(j,nclass):
	"""离散数据进行one-hot"""
	e = np.zeros((nclass, 1))
	e[j] = 1.0
        return e

    def get_format_data(X,y,isTest):
	ndim = X.shape[1]
	nclass = len(np.unique(y))
	inputs = [np.reshape(x, (ndim, 1)) for x in X]
	if not isTest:
	    results = [vectorized_result(y,nclass) for y in y]
	else:
	    results = y
	data = zip(inputs, results)
	return data

    #随机生成数据
    np.random.seed(0)
    X, y = make_moons(200, noise=0.20)
    ndim = X.shape[1]
    nclass = len(np.unique(y))
    
    #划分训练、测试集
    from sklearn.cross_validation import train_test_split
    train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=0)
    
    training_data = get_format_data(train_x,train_y,False)
    test_data = get_format_data(test_x,test_y,True)
    
    net = Network(sizes=[ndim,30,nclass])
    net.SGD(training_data=training_data,epochs=10,mini_batch_size=10,eta=0.1,test_data=test_data)

if __name__ == '__main__':
    main()
