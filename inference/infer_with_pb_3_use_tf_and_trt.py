#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ver 2: use trt from tensorflow.contrib.tensorrt for compress model

import sys
import os
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
import timer

use_hub_model = False

if use_hub_model:
	FROZEN_FPATH = '/home/andrei/Data/Datasets/Scales/pb/output_graph.pb'
	ENGINE_FPATH = '/home/andrei/Data/Datasets/Scales/pb/hub_model_engine.plan'
	INPUT_NODE = 'Placeholder-x'
	OUTPUT_NODE = 'final_result'
	INPUT_SIZE = [3, 299, 299]
else:
	#FROZEN_FPATH = '/root/tmp/saved_model_inception_resnet.pb'
	#ENGINE_FPATH = '/root/tmp/engine.plan'
	FROZEN_FPATH = 'saved_model_full_2.pb'
	ENGINE_FPATH = 'saved_model_full_2.plan'
	INPUT_NODE = 'Placeholder-x'
	OUTPUT_NODE = 'sigmoid_out'
	INPUT_SIZE = [3, 299, 299]


def get_image_as_array(image_path):

	# Read the image & get statstics
	image = Image.open(image_file)
	#img.show()
	width, height = image.size
	print(width)
	print(height)
	shape = [299, 299]
	#image = tf.image.resize_images(img, shape, method=tf.image.ResizeMethod.BICUBIC)
	image = image.resize(shape, Image.ANTIALIAS)
	image_arr = np.array(image, dtype=np.float32) / 256.0

	return image_arr

	#img_in = scipy.misc.imread(image_path, mode='RGB')	
	#img = img_in.astype(np.float32)
	#img = utils.resize_and_crop(img, CROP_SIZE)
	#img = img.transpose(2, 0, 1)
	#return img


def inference_with_graph(image, graph_def):

	#Plot the image
	#image.show()

	with open('labels.txt') as f:
		labels = f.readlines()
		labels = [x.strip() for x in labels]
		print(labels)
	#sys.exit(0)

	tf.logging.info('Starting execution.')
	tf.reset_default_graph()
	g = tf.Graph()

	input_node_0 = 'Placeholder-x:0'
	output_node_0 = 'sigmoid_out:0'

	with g.as_default():

		(graph_def=graph_def, 
			input_map={input_node: input_node_0},
			return_elements=[output_node_0])

		output = return_tensors[0].outputs[0]


	with tf.Session(graph=g) as sess:

		# Load the graph in graph_def
		print("session")

		val = sess.run([output])
		print(val)



			# We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
			with gfile.FastGFile(pb_file,'rb') as f:

				#Set default graph as current graph
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				#sess.graph.as_default() #new line

				# Import a graph_def into the current default Graph
				use_hub_model = True
				if use_hub_model:
					input_output_placeholders = ['Placeholder-x:0', 'sigmoid_out:0']
					#input_output_placeholders = ['Placeholder:0', 'final_result:0']
				else:
					input_output_placeholders = ['Placeholder-x:0', 'sigmoid_out:0']
					#input_output_placeholders = ['Placeholder-x:0', 'reluF1:0']
					#input_output_placeholders = ['Placeholder-x:0', 'reluF2:0']
					#input_output_placeholders = ['Placeholder-x:0', 'Mean:0']

				print("import graph")	
				input_, predictions =  tf.import_graph_def(graph_def, name='', 
					return_elements=input_output_placeholders)

				timer.timer('predictions.eval')
				p_val = predictions.eval(feed_dict={input_: [image_arr]})
				index = np.argmax(p_val)
				label = labels[index]
				#print(p_val)
				#print(np.max(p_val))
				#print('index={0}, label={1}'.format(index, label))
				timer.timer()

				return label


def createParser ():
	"""ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default="images/1.jpg", type=str,\
		help='input')
	parser.add_argument('-pb', '--pb', default="saved_model_full.pb", type=str,\
		help='input')
	parser.add_argument('-o', '--output', default="logs/1/", type=str,\
		help='output')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	#pb_file_name = 'saved_model.pb' # output_graph.pb
	label = inference(arguments.input, arguments.pb)
	print(label)