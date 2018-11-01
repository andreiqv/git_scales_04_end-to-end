#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ver 1: just use pb-model for inference.

import sys
import os
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
import timer


def inference(image_file, pb_file):

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

	#Plot the image
	#image.show()

	with open('labels.txt') as f:
		labels = f.readlines()
		labels = [x.strip() for x in labels]
		print(labels)
	#sys.exit(0)

	with tf.Graph().as_default() as graph:

		with tf.Session() as sess:

			# Load the graph in graph_def
			print("session")

			# We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
			with gfile.FastGFile(pb_file,'rb') as f:

				#Set default graph as current graph
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				#sess.graph.as_default() #new line

				# Import a graph_def into the current default Graph
				use_hub_model = True
				if use_hub_model:
					#input_output_placeholders = ['Placeholder-x:0', 'sigmoid_out:0']
					input_output_placeholders = ['input:0', 'output:0']
					#input_output_placeholders = ['Placeholder:0', 'final_result:0']
				else:
					input_output_placeholders = ['input:0', 'output:0']
					#input_output_placeholders = ['Placeholder-x:0', 'sigmoid_out:0']
					#input_output_placeholders = ['Placeholder-x:0', 'reluF1:0']
					#input_output_placeholders = ['Placeholder-x:0', 'reluF2:0']
					#input_output_placeholders = ['Placeholder-x:0', 'Mean:0']

				print("import graph")	
				input_, predictions =  tf.import_graph_def(graph_def, name='', 
					return_elements=input_output_placeholders)

				timer.timer('----')

				print("predictions.eval")
				p_val = predictions.eval(feed_dict={input_: [image_arr]})
				index = np.argmax(p_val)
				label = labels[index]
				#print(p_val)
				#print(np.max(p_val))
				#print('index={0}, label={1}'.format(index, label))
				timer.timer()


				print("predictions.eval")
				p_val = predictions.eval(feed_dict={input_: [image_arr]})
				index = np.argmax(p_val)
				label = labels[index]
				timer.timer()

				print("predictions.eval")
				p_val = predictions.eval(feed_dict={input_: [image_arr]})
				index = np.argmax(p_val)
				label = labels[index]
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