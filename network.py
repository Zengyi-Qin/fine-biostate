import tensorflow as tf
import numpy as np
import os
from functools import reduce


class SurfaceNet(object):
    def __init__(self, training):
        self.bn_epsilon = 0.001
        self.training = training
        return

    def build(self, x):
        mean_x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
        x = x - mean_x
        mean_absx = tf.reduce_mean(tf.abs(x), axis=[1, 2], keep_dims=True)
        x = x / (1e-9 + mean_absx)

        with tf.variable_scope('feature_space'):
            # start block
            self.convSTL = self.conv_layer(x, 7, 2, 1, 128, 'convST')
            self.poolSTL_S4 = self.avg_pool(self.convSTL, 3, 2, 'poolSTL_S4')

            # block A
            self.convAL1 = self.central_pool(self.conv_layer(
                self.poolSTL_S4, 3, 2, 128, 256, 'convA1', True, True))
            self.convAL2 = self.conv_layer(
                self.convAL1, 1, 1, 256, 64, 'convA2', False, False)
            self.convAL3 = self.conv_layer(
                self.convAL2, 3, 1, 64, 256, 'convA3')
            self.poolAL_S16 = self.avg_pool(self.convAL3, 3, 2, 'poolAL_S16')

            # block B
            self.convBL1 = self.central_pool(self.conv_layer(
                self.poolAL_S16, 3, 1, 256, 256, 'convB1', True, True))
            self.convBL2 = self.conv_layer(self.convBL1, 1, 1, 256, 64, 'convB2', False, False) + \
                self.avg_pool(self.convAL2, 3, 2, 'poolBL2')
            self.convBL3 = self.conv_layer(
                self.convBL2, 3, 1, 64, 256, 'convB3')
            self.poolBL_S32 = self.avg_pool(self.convBL3, 3, 2, 'poolBL_S32')

            # block C
            self.convCL1 = self.central_pool(self.conv_layer(
                self.poolBL_S32, 3, 1, 256, 256, 'convC1', True, True))
            self.convCL2 = self.conv_layer(self.convCL1, 1, 1, 256, 64, 'convC2', False, False) + \
                self.avg_pool(self.convBL2, 3, 2, 'poolCL2')
            self.convCL3 = self.conv_layer(
                self.convCL2, 3, 1, 64, 256, 'convC3')
            self.poolCL_S64 = self.max_pool(self.convCL3, 'poolCL_S64')

            self.convGL1 = self.conv_layer(
                self.poolCL_S64, 3, 1, 256, 256, 'convG1')
            self.convGL2 = self.conv_layer(
                self.convGL1, 3, 1, 256, 256, 'convG2')
            self.convGL3 = self.conv_layer(
                self.convGL2, 3, 1, 256, 1, 'convG3', False, False)
            self.output_regression = tf.reshape(
                tf.reduce_mean(self.convGL3, axis=1), [-1])

    def avg_pool(self, bottom, ksize, stride, name):
        return tf.nn.avg_pool(bottom, ksize=[1, ksize, 1, 1], strides=[1, stride, 1, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name=name)

    def central_pool(self, bottom):
        pool_1 = tf.nn.avg_pool(bottom, ksize=[1, 2, 1, 1], strides=[
                                1, 1, 1, 1], padding='SAME')
        pool_2 = tf.nn.avg_pool(pool_1, ksize=[1, 2, 1, 1], strides=[
                                1, 1, 1, 1], padding='SAME')
        return pool_2

    def conv_layer(self, bottom, ksize, stride, in_channels, out_channels, name, batch_norm=True, relu=True):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                ksize, in_channels, out_channels, name)
            conv = tf.nn.conv2d(
                bottom, filt, [1, stride, 1, 1], padding='SAME')
            if batch_norm:
                conv = self.group_norm(conv)
            if relu:
                conv = tf.nn.relu(conv)
            return conv

    def fc_layer(self, bottom, in_size, out_size, name, dropout=False, reg=1e-2):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, reg)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            if dropout:
                fc = tf.layers.dropout(fc, rate=0.5, training=self.training)
            return fc

    def lstm_layer(self, bottom, time_steps, in_size, out_size, name):
        with tf.variable_scope(name):
            cell = tf.nn.rnn_cell.BasicLSTMCell(out_size)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell, output_keep_prob=0.5)
            lstm_outputs, state = tf.nn.dynamic_rnn(cell, tf.reshape(
                bottom, [-1, time_steps, in_size]), dtype=tf.float32)
            lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
            lstm_last = tf.gather(lstm_outputs, int(
                lstm_outputs.get_shape()[0]) - 1)
            return tf.reshape(lstm_last, [-1, out_size, 1, 1])

    def batch_normalization_layer(self, input_layer, dimension):
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(
            input_layer, mean, variance, beta, gamma, self.bn_epsilon)

        return bn_layer

    def group_norm(self, x, G=32, esp=1e-5):
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.Variable(tf.constant(
            1.0, shape=[C]), dtype=tf.float32, name='gamma')
        beta = tf.Variable(tf.constant(
            0.0, shape=[C]), dtype=tf.float32, name='beta')
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
        return output

    def get_conv_var(self, filter_size, in_channels, out_channels, name, reg=1e-2):
        shape = [filter_size, 1, in_channels, out_channels]
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(reg)

        filters = tf.get_variable(
            name=name+'filters', shape=shape, initializer=initializer, regularizer=regularizer)
        biases = tf.get_variable(
            name=name+'biases', shape=[out_channels], initializer=initializer, regularizer=regularizer)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, reg):
        shape = [in_size, out_size]
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(reg)

        weights = tf.get_variable(
            name=name+'weights', shape=shape, initializer=initializer, regularizer=regularizer)
        biases = tf.get_variable(
            name=name+'biases', shape=[out_size], initializer=initializer, regularizer=regularizer)

        return weights, biases
