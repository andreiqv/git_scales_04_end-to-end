import math
import os
import os.path
import sys
from random import randint
import argparse
from PIL import Image, ImageDraw
import numpy as np

DATASET_DIR = '/home/andrei/Data/Datasets/Scales/classifier_dataset_181018/'

R, G, B = (0, 1, 2)
THRESHOLD = 25 # 1 - 50

def delete_red_lines(infile, outfile):
	""" Detect and delete red linies on borders of a picture.
	Inputs
	------
	infile and outfile : str, full path to input and output files
	"""

	#print(infile)
	file_basename = os.path.basename(infile)
	#file_name = ''.join(in_file.split('.')[:-1])	

	img = Image.open(infile)
	sx, sy = img.size
	#print('pixel(0,0):', img.getpixel((0, 0)))

	arr = np.asarray(img) 
	#print(arr[0,0]) # [255   2   0],  <class 'numpy.ndarray'>	
	#print(np.mean(arr[:,0,R]))
	#print(np.mean(arr[:,0,G]))
	#print(np.mean(arr[:,0,B]))
	"""
	253.94020356234097
	0.772264631043257
	0.05343511450381679
	"""

	delta = THRESHOLD
	x = 0
	while True:
		meanR = np.mean(arr[x,x,R])
		meanG = np.mean(arr[x,x,G])
		meanB = np.mean(arr[x,x,B])
		#print('x={}: mean=({:.2f},{:.2f},{:.2f})'.format(x, meanR, meanG, meanB))
		if meanR > 256-delta and meanG < delta and meanB < delta and x < math.floor(sx/2 - 1):
			x += 1
		else:
			break

	if x > 0:
		d = x
		area = (d, d, sx-d, sy-d)
		print('d={}, {}'.format(d, file_basename))
		#print('infile={}, d = {}. Crop area {} from image ({},{})'.format(file_basename, d, area, sx, sy))
		crop_box = img.crop(area)
		#out_file = outfile + '_crop.jpg'
		crop_box.save(outfile)


def process_subdir(subdir):

	files = os.listdir(subdir)	
	for filename in files:
		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]
		if not ext in {'.jpg', ".png"} : continue
		filepath = subdir + '/' + filename		
		delete_red_lines(infile=filepath, outfile=filepath)


def process_all_subdirs(src_dir):

	subdirs = os.listdir(src_dir)
	
	for subdir_name in subdirs:
		subdir_path = src_dir + '/' + subdir_name
		process_subdir(subdir_path)	


"""
def rotate_images_with_angles(in_dir, out_dir, file_names, angles):

	for i, file_name in enumerate(file_names):

		in_file_path = in_dir + '/' + file_name
		out_file_path = out_dir + '/' + file_name
		img = Image.open(in_file_path)
		angle = -angles[i]
		img_rot = img.rotate(angle)
		img_rot.save(out_file_path)
		print('{0}: {1} - angle={2}'.format(i, file_name, angle))
"""

#-----------------------------------

def createParser ():
	"""	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--src_dir', default=None, type=str,\
		help='input dir')
	return parser

if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])		

	src_dir = arguments.src_dir if arguments.src_dir else DATASET_DIR
	src_dir = src_dir.rstrip('/')
	process_all_subdirs(src_dir)