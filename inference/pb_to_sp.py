"""
tensorboard --logdir=output/
"""

import argparse
import sys

import tensorflow as tf
from tensorflow.python.platform import gfile


def convert(in_file, out_dir):

	with tf.Session() as sess:
		#model_filename ='output_graph.pb'
		model_filename = in_file
		with gfile.FastGFile(model_filename, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			g_in = tf.import_graph_def(graph_def)

	LOGDIR = out_dir
	train_writer = tf.summary.FileWriter(LOGDIR)
	train_writer.add_graph(sess.graph)



def createParser ():
	"""ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default="saved_model.pb", type=str,\
		help='input')
	parser.add_argument('-o', '--output', default="logs/1/", type=str,\
		help='output')
	return parser

if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	convert(arguments.input, arguments.output)