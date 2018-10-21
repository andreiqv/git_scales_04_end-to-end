#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
sys.path.append('..')
import os.path
import neural_networks.networks as networks

USE_HUB = not os.path.exists('.dont_use_hub')
use_hub = USE_HUB


if USE_HUB:

	import tensorflow_hub as hub
	model_number = 3
	type_model = 'feature_vector'
	#type_model = 'classification'
	
	if model_number == 1:		
		sp_model = "https://tfhub.dev/google/imagenet/resnet_v2_152/{0}/1".format(type_model)
		SHAPE = 224, 224, 3
		bottleneck_tensor_size =  1536
	elif model_number == 2:
		sp_model = "https://tfhub.dev/google/imagenet/inception_v3/{0}/1".format(type_model)
		SHAPE = 299, 299, 3
		bottleneck_tensor_size =  1536
	elif model_number == 3:
		sp_model = "https://tfhub.dev/google/imagenet/inception_resnet_v2/{0}/1".format(type_model)
		SHAPE = 299, 299, 3
		bottleneck_tensor_size =  1536
	else:
		raise Exception('Bad n_model')
		# https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1

	hub_module = lambda trainable=False: hub.Module(sp_model)


else:  # for local testing
	network_model = networks.conv_network_224
	SHAPE = 224, 224, 3
	bottleneck_tensor_size =  588 #1536
