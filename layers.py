#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
import sys
import math
import numpy as np


def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',  name=name) 

def max_pool_3x3(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME',  name=name)

#-------------

def convPoolLayer(p_in, kernel, pool_size, num_in, num_out, func=None, name=''):
	
	W = weight_variable([kernel[0], kernel[1], num_in, num_out], name='W'+name)  # 32 features, 5x5
	b = bias_variable([num_out], name='b'+name)

	h = conv2d(p_in, W, name='conv'+name) + b

	if func:
		h = func(h, name='relu'+name)

	if pool_size == 1:
		p_out = h
	elif pool_size == 2:
		p_out = max_pool_2x2(h, name='pool'+name)
	elif pool_size == 3:
		p_out = max_pool_3x3(h, name='pool'+name)
	else:
		raise("the pool size = {0} is not supported".format(pool_size))
	
	print('p{0} = {1}'.format(name, p_out))

	return p_out

def fullyConnectedLayer(p_in, input_size, num_neurons, func=None, name=''):
	#num_neurons_6 = 128
	W = weight_variable([input_size, num_neurons], name='W'+name)
	b = bias_variable([num_neurons], name='b'+name)
	h = tf.matmul(p_in, W) + b

	if func:
		h = func(h, name=func.__name__ + name)

	print('h{0} = {1}'.format(name, h))
	return h


