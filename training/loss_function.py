#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
25.10.2018

"""
from __future__ import absolute_import,  division, print_function
import os
import sys
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import tensorflow as tf

#NUM_CLASSES = 11
NUM_CLASSES = 8

def sigma6(v, size_v):
	"""the sum of all products of k=6 distinct elements of b
	"""
	#size_v = v.shape[0]  # = the number of classes
	#size_v = NUM_CLASSES - 1
	k = 6
	sum6 = 0
	for i1 in range(0, size_v-k+1):
		for i2 in range(i1+1, size_v-k+2):
			for i3 in range(i2+1, size_v-k+3):
				for i4 in range(i3+1, size_v-k+4):
					for i5 in range(i4+1, size_v-k+5):
						for i6 in range(i5+1, size_v-k+6):
							sum6 += v[i1] * v[i2] * v[i3] * v[i4] * v[i5] * v[i6]	
	return sum6
	
def sigma5(v, size_v):
	"""the sum of all products of k=5 distinct elements of b
	"""
	#size_v = len(v)
	#size_v = v.shape[0] 
	#size_v = NUM_CLASSES - 1
	k = 5
	sum5 = 0
	for i1 in range(0, size_v-k+1):
		for i2 in range(i1+1, size_v-k+2):
			for i3 in range(i2+1, size_v-k+3):
				for i4 in range(i3+1, size_v-k+4):
					for i5 in range(i4+1, size_v-k+5):
						sum5 += v[i1] * v[i2] * v[i3] * v[i4] * v[i5]
						print('i1={}, i2={}, i3={}, i4={}, i5={}, sum5={}'.\
							format(i1, i2, i3, i4, i5, sum5))
	return sum5


def sigma(k, v, size_v):

	t = 0
	


def loss_function_top_6(network_output, label_one_hot, vector_size, k=6, tau=0.1):
	"""  Smooth Loss Functions for Deep Top-k Classification
	"""	
	k = tf.constant(k, dtype=tf.float32) # tf.int64 doesn't work
	tau = tf.constant(tau, dtype=tf.float32)

	s = tf.reshape(network_output, [vector_size])  # it's usually softmax output
	y = tf.argmax(tf.reshape(label_one_hot, [vector_size]))  # index of label (tf.int64)
	y = tf.cast(y, tf.int32)

	sy = s[y]
	
	s_without_y = tf.concat([s[:y], s[y+1:]], 0)

	#s_without_y = tf.cond(y < NUM_CLASSES-1, 
	#	lambda: tf.concat([s[:y], s[y+1:]], 0),
	#	lambda: s1)

	#s1 = s[:y]
	#s2 = s[y+1:]
	#s_without_y = tf.concat([s1,s2], 0)	
	
	a = tf.exp(sy / (k*tau))
	b = tf.exp(s_without_y / (k*tau))
	size_b = vector_size - 1

	#loss = tau * tf.log( 1 + tf.exp(1/tau) * 1/a * sigmak(k, b) / sigmak(k-1, b) )
	#loss = tau * tf.log( 1 + tf.exp(1/tau) * 1/a \
	#	* sigma6(b, size_b) / sigma5(b, size_b) )
	
	return loss


if __name__ == '__main__':

	
	s = tf.constant([0.1, 0.9, 0.0, 0.3, 0.4, 0.5, 0.3, 0.2, 0.1, 0.3, 0.4])
	l = tf.constant([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	num_classes = 11
	#s = tf.constant([0.1, 0.9, 0.0, 0.3, 0.4, 0.5, 0.3, 0.2])
	#s = tf.constant([0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0])
	#s = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
	#s = tf.constant([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	#l = tf.constant([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	loss = loss_function_top_6(s, l, vector_size=num_classes, tau=0.01)

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		loss_val = loss.eval(session=sess)

		print(loss_val)

	

	#b = [0,1,2,3,4,5]
	#sigma5(b)