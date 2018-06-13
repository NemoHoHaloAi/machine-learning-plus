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

def visual(neurons):
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
    ax.plot(XX, YY, 'b*')

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
    lines = []
    for i in range(len(sxs)):
        lines.append(ax.plot([sxs[i],exs[i]],[sys[i],eys[i]],'-',linewidth=1.))

    def animate(i):
        tmp = []
        for line in lines:
            for l in line:
                l.set_linewidth(np.random.randn())
                tmp.append(l)
        return tmp
    anim = animation.FuncAnimation(fig, animate, frames=200, interval=100, blit=True)
    plt.show()


def main():
    """
    程序入口
    """
    enviroment_init()
    visual([4,9,1])

if __name__ == '__main__':
    main()
