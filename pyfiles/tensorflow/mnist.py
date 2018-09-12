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

"""
第一版：
    模型：1层的神经网络
    权重、bias：全0初始化
    激活函数：Softmax
    损失函数：交叉熵
    优化方式：梯度下降
    准确率：训练1000次，下降步长0.01，测试集得分0.79

存在问题：
    0. 模型过于简单，抽象能力不够
    1. 权重、bias的初始值不太合理

可优化空间：
    0. 权重加入一点噪声避免对称性以及0梯度等问题
    1. 激活函数使用ReLU，因此bias最好使用一个小的正数来初始化，以避免神经元节点输出恒为0的问题
    2. 增加层数提高抽象能力
    3. 卷积、池化
"""

def enviroment_init():
    """
    程序运行环境初始化
    """
    pass

def init_weights(shape):
    """
    优化1
    初始化权重，使用truncated_normal方法随机正态分布取值
    随机值作为噪声，避免出现对称性以及0梯度问题
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def init_bias(shape):
    """
    优化2
    初始化bias，不用0，而是使用一个小的正数0.1来避免节点输出恒为0的问题
    """
    return tf.Variable(tf.constant(.1, shape=shape))

def conv2d(x, W):
    """
    优化3
    做卷积处理，边距选择SAME，步长为1，保证输入输出一样大
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    优化4
    最大池化，使用传统2x2模板
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def nn_opt():
    """
    对最初的nn进行优化：1.权重、bias，2.网络结构，3.加入卷积、池化、dropout等手段
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder('float32', [None, 28*28])
    """
    第一层卷积
    现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。
    卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
    而对于每一个输出通道都有一个对应的偏置量。
    """
    W_conv1 = init_weights([5, 5, 1, 32]) # 输入通道是1，也就是灰度图
    b_conv1 = init_bias([32]) # 对应输出通道的bias值
    """
    为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    (因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
    """
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # 为了能够运算，转为4d向量
    """
    我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
    """
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1) # 先对输入、权重进行卷积，然后应用ReLU激活函数
    h_pool1 = max_pool_2x2(h_conv1) # 对第一层卷积输出应用池化
    """
    第二层卷积
    为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
    """
    W_conv2 = init_weights([5, 5, 32, 64]) # 注意第二层的输入是第一层的输出，第一层的输出是32个特征，因此此处是32
    b_conv2 = init_bias([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    """
    密集连接层
    现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
    我们把池化层输出的张量reshape成一维向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
    """
    W_fc1 = init_weights([7*7*64, 1024]) # zzz
    b_fc1 = init_bias([1024]) # zzz
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # zzz
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1) # zzz
    """
    Dropout
    为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，
    还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。
    """
    keep_prob = tf.placeholder("float") # 定义placeholder来代表一个神经元的输出在dropout中保持不变的概率，方便根据是训练还是测试来动态设置值
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    """
    输出层
    最后，我们添加一个softmax层，就像前面的单层softmax regression一样。
    """
    W_fc2 = init_weights([1024, 10]) # 权重可以理解为左边是输入特征数，后边是输出特征数，因为我们的目标是获取到0~9共10个可能的概率，因此有10个输出
    b_fc2 = init_bias([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2) # 因为输出是10个可能的概率，因此选择Softmax做输出层的激活函数
    """
    测试一下效果：
    """
    y_ = tf.placeholder('float32', [None, 10])
    cross_entropy = -tf.reduce_mean(tf.mul(y_, tf.log(y_conv)))
    #使用更复杂的ADAM优化器做梯度下降
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    with tf.Session() as sess:
        sess.run(init)
        count = 10000
        print('Start nn with opt, times='+str(count)+'--')
        for i in range(count):
            if i%1000==0:
                print('Idx:'+str(i))
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, keep_prob: .5, y_: batch_ys})
        print sess.run(accuracy, feed_dict={x: mnist.test.images, keep_prob: 1., y_: mnist.test.labels})
        print('End----')

def nn():
    # 通过提供的input_data.py加载数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 构建阶段
    # 定义描述每张图片向量的变量，用一个二维浮点数张量
    x = tf.placeholder('float32', [None, 28*28])
    # 定义权重以及bias，这两个都是随机初始值的，因此指定为0也可以（不过一般还是用正态分布随机取值来避免不均匀的情况）
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
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
        count = 20000
        print('Start nn without opt, times='+str(count)+'--')
        # 训练1000次
        for i in range(count):
            # 通过next_batch获取数据块，xs为训练数据，ys为训练标签
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # 执行train_step op，利用feed指定变量x和y_的值
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # 在测试集上验证下效果
        print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print('End----')

if __name__ == '__main__':
    nn()
    nn_opt()
