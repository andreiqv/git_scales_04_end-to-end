#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
"""

import os
import os.path
import sys
import argparse
from PIL import Image, ImageDraw
import _pickle as pickle
import gzip
import random
from random import randint
import math
import numpy as np

import tensorflow as tf
#from tensorflow.contrib.data import Dataset, Iterator
Dataset = tf.data.Dataset
Iterator = tf.data.Iterator

import network
import split_data
import distort

DO_MIX = True
NUM_CLASSES = 6
CHECKPOINT_NAME = 'my_test_model'

from layers import *

if os.path.exists('.notebook'):
	bottleneck_tensor_size =  588
	BATCH_SIZE = 2
	DISPLAY_INTERVAL, NUM_ITERS = 1, 500
else:
	bottleneck_tensor_size =  1536
	#bottleneck_tensor_size =  1001
	BATCH_SIZE = 10
	DISPLAY_INTERVAL, NUM_ITERS = 100, 20*1000*1000


if os.path.exists('.notebook'):
	#data_dir = '../data'
	data_dir = '../separated'
	module = network.conv_network_224
	SHAPE = 224, 224, 3
else:
	#data_dir = '/home/chichivica/Data/Datasets/Scales/data'
	#data_dir = '/home/chichivica/Data/Datasets/Scales/separated'
	data_dir = '/home/chichivica/Data/Datasets/Scales/separated_cropped/'
	import tensorflow_hub as hub

	model_number = 3
	type_model = 'feature_vector'
	#type_model = 'classification'
	
	if model_number == 1:		
		module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/{0}/1".format(type_model))
		SHAPE = 224, 224, 3
	elif model_number == 2:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/{0}/1".format(type_model))
		SHAPE = 299, 299, 3
	elif model_number == 3:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/{0}/1".format(type_model))
		SHAPE = 299, 299, 3
	else:
		raise Exception('Bad n_model')
		# https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1

np.set_printoptions(precision=4, suppress=True)



def network1(input_tensor, input_size, output_size):

	f1 = fullyConnectedLayer(
		input_tensor, input_size=bottleneck_tensor_size, num_neurons=output_size, 
		func=tf.nn.sigmoid, name='F1') # func=tf.nn.relu
	
	return f1

neural_network = network1	

#---------------------------------


def input_parser_tf(image_path):
	# convert the label to one-hot encoding
	#NUM_CLASSES = 11
	#input_height, input_width = SHAPE[0], SHAPE[1]

	# read the img from file
	image_string = tf.read_file(image_path)
	image_decoded = tf.image.decode_jpeg(image_string)
	image_resized = tf.image.resize_images(image_decoded, [SHAPE[1], SHAPE[0]],
                                               method=tf.image.ResizeMethod.BICUBIC)
	image = tf.cast(image_resized, tf.float32) / tf.constant(255.0)

	return image


def input_parser_np(image_path, shape):	

	image_size = (shape[0], shape[1])
	im = Image.open(image_path)		
	im = im.resize(image_size, Image.ANTIALIAS)
	arr = np.array(im, dtype=np.float32) / 256

	return arr


def inference(image, shape):

	# 3) Calculate bottleneck in TF
	height, width, color =  shape

	x = tf.placeholder(tf.float32, [height, width, 3], name='Placeholder-x')
	#x1 = tf.placeholder(tf.float32, [None, bottleneck_tensor_size], name='Placeholder-x0') # Placeholder for input.
	logits = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='Placeholder-y')   # Placeholder for labels.

	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")		
	
	# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]

	input_bottleneck = tf.reshape(bottleneck_tensor, [-1, bottleneck_tensor_size])
	logits = neural_network(input_bottleneck, bottleneck_tensor_size, NUM_CLASSES)
	print('logits =', logits)

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		tf.train.Saver().restore(sess, './save_model/{0}'.format(CHECKPOINT_NAME))

		#feature_vector = bottleneck_tensor.eval(feed_dict={ x : image })

		feed_dict={ x : image }
		#feed_dict = {x1 : feature_vector}

		output_values = logits.eval(feed_dict=feed_dict)
		
		print(output_values)

	"""
	# 3) Calculate bottleneck in TF
	height, width, color =  shape
	x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")		
	
	# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	print('bottleneck_tensor:', bottleneck_tensor)			

	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.

		feature_vector = bottleneck_tensor.eval(feed_dict={ x : [arr] })			

		feature_vectors.append(feature_vector)
		labels.append(label)
		filenames.append(filename) # or file_path
	"""
#------------------------------------------------


def save_data_dump(data, dst_file):
	
	# save the data on a disk
	dump = pickle.dumps(data)
	print('dump done')
	f = gzip.open(dst_file, 'wb')
	print('gzip done')
	f.write(dump)
	print('dump was written')
	f.close()



def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--src_dir', default=None, type=str,\
		help='input dir')
	parser.add_argument('-o', '--dst_file', default=None, type=str,\
		help='output file')
	parser.add_argument('-m', '--mix', dest='mix', action='store_true')

	parser.add_argument('-n', '--num', default=100, type=int,\
		help='num_angles for a single picture')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	NUM_ANGLES 	 = arguments.num
	#DO_MIX 		 = arguments.mix
	print('NUM_ANGLES =', 	NUM_ANGLES)
	#print('DO_MIX =',		DO_MIX)

	if not arguments.src_dir:
		src_dir = data_dir
	if not arguments.dst_file:		
		dst_file = 'dump.gz'

	image_path = '../data/test/img_2_0_2018-08-26-20-49-307283_61.jpg'	
	image = input_parser_np(image_path, shape=SHAPE)	
	inference(image, shape=SHAPE)

