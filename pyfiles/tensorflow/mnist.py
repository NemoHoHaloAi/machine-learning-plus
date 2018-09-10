#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
project_name	TensorFlow-MNIST
function	使用Tensorflow实现MNIST的图片数字识别
author		Ho Loong
date		2018-09-10
company		Aispeech,Inc.
ps              Please be pythonic.
'''

import sys
import os
import input_data
import tensorflow as tf

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

    # 通过提供的input_data.py加载数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 构建阶段
    # 定义描述每张图片向量的变量，用一个二维浮点数张量
    x = tf.placeholder('float32', [None, 28*28])
    # 定义权重以及bias，这两个都是随机初始值的，因此指定为0也可以（不过一般还是用正态分布随机取值来避免不均匀的情况）
    W = tf.Variable(tf.zeros([784,10]))
    #W = tf.Variable(tf.random_normal([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    #b = tf.Variable(tf.random_normal([10]))
    # 定义Softmax模型
    y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))
    # 定义正确值的占位符
    y_ = tf.placeholder('float32', [None, 10])
    # 定义损失函数，此处使用交叉熵
    cross_entropy = -tf.reduce_mean(tf.mul(y_, tf.log(y)))
    # 定义优化op，使用梯度下降来优化参数，优化指标就是前面定义的交叉熵损失函数
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 定义初始化op
    init = tf.initialize_all_variables()
    # 定义输出准确率的op
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 执行阶段
    with tf.Session() as sess:
        # 执行变量初始化op
        sess.run(init)
        # 训练1000次
        for i in range(1000):
            # 通过next_batch获取数据块，xs为训练数据，ys为训练标签
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # 执行train_step op，利用feed指定变量x和y_的值
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # 在测试集上验证下效果
        print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

if __name__ == '__main__':
    main()
