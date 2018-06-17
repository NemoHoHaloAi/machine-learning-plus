#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
project_name	模拟神经网络BP更新权重
function	反向传播算法更新权重，使用or、and、xor示例
author		Ho Loong
date		2018-06-11
company		freeman
ps              Please be pythonic.
'''

import sys
import os
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation

def enviroment_init():
    """
    程序运行环境初始化
    """
    pass

def visual(neurons, values):
    values_line = []
    for value in values:
        values_line+=value
    """
    训练可视化
    """
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))

    def gen_neuron_pos(neurons): # 神经网络组成结构，例如一个两层网络如下：[2,3,1]，即输入层2个神经元，隐含层3个，输出层1个
        """
        生成所有神经元对应的各自位置
        """
        X = [[1./(len(neurons)+1)*(i+1)]*neurons[i] for i in range(len(neurons))]
        Y = [[1./(size+1)*(j+1) for j in range(size)] for size in neurons]
        X = [[i*(2-0)+0 for i in x] for x in X]
        Y = [[i*(2-(-2))+(-2) for i in y] for y in Y]
        XX,YY = [],[]
        for x in X:
            for z in x:
                XX.append(z)
        for y in Y:
            for z in y:
                YY.append(z)
        return X,Y,XX,YY
    X,Y,XX,YY = gen_neuron_pos(neurons)
    ax.plot(XX, YY, 'ko')

    def gen_neuron_text(XX,YY):
        """
        生成每个神经元对应的文本（数值显示）位置
        """
        X_text,Y_text = [],[]
        for i in range(len(XX)):
            X_text.append(XX[i]-abs(0.02*(2-0)))
            Y_text.append(YY[i]+abs(0.02*(2-(-2))))
        return X_text,Y_text

    X_text,Y_text = gen_neuron_text(XX,YY)
    texts = []
    for i in range(len(X_text)):
        texts.append(plt.text(X_text[i],Y_text[i],values_line[i]))

    def gen_weight_line(X,Y):
        """
        生成每个神经元与后一层所有神经元之间的对应线坐标
        """
        start_xs,start_ys,end_xs,end_ys = [],[],[],[]
        for i in range(len(X)-1): # 遍历除最外层以外的所有层，例如两层网络的话就是遍历输入层和隐含层
            x,y,next_x,next_y = X[i],Y[i],X[i+1],Y[i+1]
            for j in range(len(x)):
                for k in range(len(next_x)):
                    start_x,start_y = x[j],y[j]
                    end_x,end_y = next_x[k],next_y[k]
                    start_xs.append(start_x)
                    start_ys.append(start_y)
                    end_xs.append(end_x)
                    end_ys.append(end_y)
        return start_xs,start_ys,end_xs,end_ys
    sxs,sys,exs,eys = gen_weight_line(X,Y)
    liness = []
    for i in range(len(sxs)):
        liness.append(ax.plot([sxs[i],exs[i]],[sys[i],eys[i]],'-',linewidth=1.))

    def animate(i):
        tmp = []
        for lines in liness:
            for line in lines:
                line.set_linewidth(np.random.randn())
                tmp.append(line)
        for text in texts:
            text.set_text(str(np.random.randn()))
            tmp.append(text)
        return tmp
    # anim = animation.FuncAnimation(fig, animate, frames=200, interval=100, blit=True)
    plt.show()

class Neuron(object):
    """
    神经网络类：
        模拟神经网络对and、or、xor三种需求逻辑的bp训练过程，并逐步逼近最终目标输出的可视化过程

    过程：
        1. 初始状态：
            此时神经网络处于weight均为1的情况下，输出为各种神经元数值加权和；
        2. 指定target：
            1. 类似指定and为结果，此时原始的输出与当前的输出是有一个平均误差的，目标就是将该平均误差最小化；
            2. 方法就是利用bp+梯度下降的方式进行weight的更新来达到对输出的控制；
    
    特点：
        这种模拟神经网络中没有激活函数的概率貌似；
    """
    def __init__(self, i, weights):
        """init"""
        self.inputs = i # 一维列表，表示输入层的初始数值
        self.weights = weights # 二维列表，每个元素代表当前层每个神经元链接到下一层各个神经元的权重，例如2,3,1的网络神经元结构，则此处对应[[[w13,w14,w15],[w23,w24,w25]],[[w36],[w46],[w56]]]
        self.feed_forward()

    def feed_forward(self):
        """
        计算前向传输的输出
        """
        print str(self.inputs)+'->'+str(self.weights[0])
        # hidden layer
        self.hiddens = [0]*len(self.weights[0][0])
        for i in range(len(self.weights[0])): # 遍历每个隐含层的神经元
            for j in range(len(self.weights[0][i])): # 遍历每个神经元对应的输入层的输入权重
                weight = self.weights[0][i][j]
                self.hiddens[j] += weight*self.inputs[i]
        print str(self.hiddens)+'->'+str(self.weights[1])

        # output layer
        self.outputs = [0]*len(self.weights[1][0])
        for i in range(len(self.weights[1])): # 遍历每个输出层的神经元
            for j in range(len(self.weights[1][i])): # 遍历每个神经元对应的隐含层的输入权重
                weight = self.weights[1][i][j]
                self.outputs[j] += weight*self.hiddens[i]
        print self.outputs


    def values(self):
        """
        返回当前网络各层的数值
        """
        return self.inputs,self.hiddens,self.outputs

    def error(self, target):
        """
        计算总误差
        """
        return .5*sum([(target[i]-self.outputs[i])**2 for i in range(len(target))])/len(target)

    def update(self, target):
        """
        设置输入输出对应目标，此处简单考虑，输入默认为(0,0)、(0,1)、(1,0)、(1,1)四种，因此该target只需要为一个长度为4的list即可，且由0,1组成
        循环更新权重，直至到一个可接受的值：
            1. 计算总误差，也就是每个输出的误差和
                公式：sum(1/2*(target-now)^2)
            2. 隐含层---->输出层的权值更新
                1. 以权重参数w36为例，如果我们想知道w36对整体误差error产生了多少影响，可以用整体误差对w36求偏导求出：（使用链式法则）
        """
        print 'Average Error:'+str(self.error(target))

def main():
    """
    程序入口
    """
    enviroment_init()
    neuron = Neuron([1,1],[[[1,1,1],[1,1,1]],[[1],[1],[1]]])
    neuron.update([1])
    #visual([2,3,1],neuron.values())

if __name__ == '__main__':
    main()
