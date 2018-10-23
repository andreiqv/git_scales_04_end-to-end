import os

DO_MIX = False
DO_AUGMENTATION = True
MULT = 5 # how many times to repeat

#NUM_CLASSES = 0
CHECKPOINT_NAME = 'full_model'
PB_FILE_NAME = 'saved_model_full_trainable.pb'
LABELS_FILE = 'labels.txt'

INPUT_NODE_NAME = 'input'
OUTPUT_NODE_NAME = 'softmax'   # the final node name of networks
#OUTPUT_NODE = 'softmax'

LEARNING_RATE = 0.01   # for last layer training 
LEARNING_RATE_FULL_MODEL = 0.001  
DO_BALANCING = False 
# if very few images in directory then use it a few times (in train dataset)

#--------

USE_HUB = not os.path.exists('.dont_use_hub')
use_hub = USE_HUB


if USE_HUB:

	DATASET_DIR= '/home/andrei/Data/Datasets/Scales/classifier_dataset_181018/'	
	BATCH_SIZE = 32  # batch for training last layer on bottleneck
	DATASET_BATCH_SIZE = 64 # batch for creating bottleneck or training full network

	NUM_ITERS = 1000*100  # the total num of iterations for training the last layer
	NUM_ITERS_CHECKPOINT = 1000*100
	NUM_ITERS_DISPLAY = 1000  # num of iterations between evaluation of valid acc and display it

	NUM_EPOCH_FULL_MODEL = 100 # the num of epoch for training full network
	NUM_ITERS_DISPLAY_FULL_MODEL = 10
	#DISPLAY_INTERVAL_FULL_MODEL = 1

else:  # for local testing
	
	DATASET_DIR = '../../separated'
	BATCH_SIZE = 3 
	DATASET_BATCH_SIZE = 16

	NUM_ITERS = 30
	NUM_ITERS_CHECKPOINT = 1000*1
	NUM_ITERS_DISPLAY = 1

	NUM_EPOCH_FULL_MODEL = 5
	NUM_ITERS_DISPLAY_FULL_MODEL = 1


#-------

DATASET_DIR = DATASET_DIR.rstrip('/')	