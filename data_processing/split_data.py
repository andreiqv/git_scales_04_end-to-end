import random
import sys
import numpy as np

def split_data_v5(data, ratio, do_balancing=False):
	""" ver-5:
	в пределах каждой папки (категории товаров) проводится сортировка по именам файлов, 
	и первые файлы в списке попадают в train, а остальные в valid.
	Затем в train и valid идет независимое перемешивание данных.
	"""

	zip3 = list(zip(data['labels'], data['images'], data['filenames']))

	#random.shuffle(zip3)
	#print('mix ok')

	# divide by categories
	labels = { z[0] for z in zip3 }
	category = dict()
	for label in labels:
		category[label] = [z for z in zip3 if z[0] == label]
		category[label].sort(key = lambda z : z[2])
		for item in category[label]:
			print('{0}: {1}'.format(item[0], item[2]))

	#sys.exit(0)		

	szip = {'train': [], 'valid': [], 'test': []}	# splitted zip

	for label in labels:
		len_data = len(category[label])
		len_valid = len_data * ratio[1] // sum(ratio)
		len_test  = len_data * ratio[2] // sum(ratio)
		len_train = len_data - len_valid - len_test	 # all rest in train set
		#if 'len_train = len_data * ratio[0] // sum(ratio)', then not all images will be used
		
		# balancing
		koef_mult = 1
		if do_balancing: # 
			if len_train >= 100 and len_train < 300:
				koef_mult = 300 // len_train
			elif len_train < 100:
				koef_mult = 100 // len_train			

		szip['train'] += category[label][ : len_train] * koef_mult
		szip['valid'] += category[label][len_train : len_train + len_valid]
		szip['test']  += category[label][len_train + len_valid : ]

		print('Label {0}: {1} images [{2} {3} {4}]'.format(label, len_data, len_train, len_valid, len_test))
		print(szip['train'][-1])

	# shuffle
	random.shuffle(szip['train'])
	random.shuffle(szip['valid'])

	"""
	randomize = np.arange(len(szip['train']))
	np.random.shuffle(randomize)
	szip['train'] = szip['train'][randomize]

	randomize = np.arange(len(szip['valid']))
	np.random.shuffle(randomize)
	szip['valid'] = szip['valid'][randomize]
	"""

	# divided on separeted lists: labels, images, filenames.
	sdata = {'train': dict(), 'valid': dict(), 'test': dict()} # splitted dataset

	for key in sdata:
		sdata[key]['labels']    = [x[0] for x in szip[key]]
		sdata[key]['images']    = [x[1] for x in szip[key]]
		sdata[key]['filenames'] = [x[2] for x in szip[key]]	

	print('train:{}, valid:{}, test:{}'.\
		format(len(sdata['train']['labels']), len(sdata['valid']['labels']), len(sdata['test']['labels'])))

	for key in sdata:
		sdata[key]['size'] = len(sdata[key]['labels'])

	return sdata	



def split_data_v4(data, ratio, do_balancing=False):
	""" ver-4:
	Shuffle and split dataset in train, valid and test subsets 
	for each category.
	Add expansion of small categories.
	"""

	zip3 = list(zip(data['labels'], data['images'], data['filenames']))

	#random.shuffle(zip3)
	print('mix ok')

	# divide by categories
	labels = { z[0] for z in zip3 }
	category = dict()
	for label in labels:
		category[label] = [z for z in zip3 if z[0] == label]

	szip = {'train': [], 'valid': [], 'test': []}	# splitted zip

	for label in labels:
		len_data = len(category[label])
		len_valid = len_data * ratio[1] // sum(ratio)
		len_test  = len_data * ratio[2] // sum(ratio)
		len_train = len_data - len_valid - len_test	 # all rest in train set
		#if 'len_train = len_data * ratio[0] // sum(ratio)', then not all images will be used
		
		koef_mult = 1
		if do_balancing:
			if len_train >= 100 and len_train < 300:
				koef_mult = 300 // len_train
			elif len_train < 100:
				koef_mult = 100 // len_train			

		szip['train'] += category[label][ : len_train] * koef_mult
		szip['valid'] += category[label][len_train : len_train + len_valid]
		szip['test']  += category[label][len_train + len_valid : ]

		print('Label {0}: {1} images [{2} {3} {4}]'.format(label, len_data, len_train, len_valid, len_test))
		print(szip['train'][-1])

	#random.shuffle(szip['train'])	

	sdata = {'train': dict(), 'valid': dict(), 'test': dict()} # splitted dataset

	for key in sdata:
		sdata[key]['labels']    = [x[0] for x in szip[key]]
		sdata[key]['images']    = [x[1] for x in szip[key]]
		sdata[key]['filenames'] = [x[2] for x in szip[key]]	

	print('train:{}, valid:{}, test:{}'.\
		format(len(sdata['train']['labels']), len(sdata['valid']['labels']), len(sdata['test']['labels'])))

	for key in sdata:
		sdata[key]['size'] = len(sdata[key]['labels'])

	return sdata	


def split_data_v3(data, ratio):
	""" ver-3:
	Shuffle and split dataset in train, valid and test subsets 
	for each category.
	It means 
	"""

	zip3 = list(zip(data['labels'], data['images'], data['filenames']))

	random.shuffle(zip3)
	print('mix ok')

	# divide into classes
	labels = { z[0] for z in zip3 }
	category = dict()
	for label in labels:
		category[label] = [z for z in zip3 if z[0] == label]

	szip = {'train': [], 'valid': [], 'test': []}	# splitted zip

	for label in labels:
		len_data = len(category[label])
		len_valid = len_data * ratio[1] // sum(ratio)
		len_test  = len_data * ratio[2] // sum(ratio)
		len_train = len_data - len_valid - len_test	 # all rest in train set
		#if 'len_train = len_data * ratio[0] // sum(ratio)', then not all images will be used

		szip['train'] += category[label][ : len_train] * koef_mult
		szip['valid'] += category[label][len_train : len_train + len_valid]
		szip['test']  += category[label][len_train + len_valid : ]

		print('Label {0}: {1} images [{2} {3} {4}]'.format(label, len_data, len_train, len_valid, len_test))
		print(szip['train'][-1])

	#random.shuffle(szip['train'])	

	sdata = {'train': dict(), 'valid': dict(), 'test': dict()} # splitted dataset

	for key in sdata:
		sdata[key]['labels']    = [x[0] for x in szip[key]]
		sdata[key]['images']    = [x[1] for x in szip[key]]
		sdata[key]['filenames'] = [x[2] for x in szip[key]]	

	print('train:{}, valid:{}, test:{}'.\
		format(len(sdata['train']['labels']), len(sdata['valid']['labels']), len(sdata['test']['labels'])))

	for key in sdata:
		sdata[key]['size'] = len(sdata[key]['labels'])

	return sdata	


def split_data_v2(data, ratio, num_labels):
	""" Split and check that train set contains representatives of all classes.
	"""

	len_data = len(data['labels'])
	assert len_data == len(data['labels'])

	#len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)
	len_train = len_data - len_valid - len_test	 # all rest in train set
	print(len_train, len_valid, len_test)

	splited_data = {'train': dict(), 'valid': dict(), 'test': dict()}
	
	for key in data:
		splited_data['train'][key] = data[key][ : len_train]
		splited_data['valid'][key] = data[key][len_train : len_train + len_valid]
		splited_data['test'][key] = data[key][len_train + len_valid : ]

	train_labels = set(splited_data['train']['labels'])
	valid_labels = set(splited_data['valid']['labels'])
	test_labels = set(splited_data['test']['labels'])

	print('\n train_labels:', train_labels)
	print('\n valid_labels:', valid_labels)
	print('\n test_labels:', test_labels)
	
	if len(valid_labels - train_labels) > 0:
		print('No labels {0} in train data but in valid'.format(valid_labels - train_labels))
	if len(test_labels - train_labels) > 0:
		print('No labels {0} in train data but in test'.format(test_labels - train_labels))


	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])

	return splited_data	


def split_data_v1(data, ratio):
	""" Split data in train, valid and test datasets.
	"""

	len_data = len(data['labels'])
	assert len_data == len(data['labels'])
	#len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)
	len_train = len_data - len_valid - len_test	 # all rest in train set
	print(len_train, len_valid, len_test)

	splited_data = {'train': dict(), 'valid': dict(), 'test': dict()}
	
	for key in data:
		splited_data['train'][key] = data[key][ : len_train]
		splited_data['valid'][key] = data[key][len_train : len_train + len_valid]
		splited_data['test'][key] = data[key][len_train + len_valid : ]

	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])

	return splited_data

split_data = split_data_v1



"""
def split_data_v4(data, ratio):
	zip3 = list(zip(data['labels'], data['images'], data['filenames']))
	#random.shuffle(zip3)
	#print('mix ok')

	# divide into classes
	labels = { z[0] for z in zip3 }
	category = dict()
	for label in labels:
		category[label] = [z for z in zip3 if z[0] == label]

	szip = {'train': [], 'valid': [], 'test': []}	# splitted zip

	for label in labels:
		len_data = len(category[label])
		len_valid = len_data * ratio[1] // sum(ratio)
		len_test  = len_data * ratio[2] // sum(ratio)
		len_train = len_data - len_valid - len_test	 # all rest in train set
		#if 'len_train = len_data * ratio[0] // sum(ratio)', then not all images will be used
		szip['train'] += category[label][ : len_train]
		szip['valid'] += category[label][len_train : len_train + len_valid]
		szip['test']  += category[label][len_train + len_valid : ]

		print('Label {0}: {1} images [{2} {3} {4}]'.format(label, len_data, len_train, len_valid, len_test))
		print(szip['train'][-1])

	random.shuffle(szip['train'])	

	sdata = {'train': dict(), 'valid': dict(), 'test': dict()} # splitted dataset

	sdata['train']['labels']    = [x[0] for x in szip['train']]
	sdata['train']['images']    = [x[1] for x in szip['train']]
	sdata['train']['filenames'] = [x[2] for x in szip['train']]	

	sdata['valid']['labels']    = [x[0] for x in szip['valid']]
	sdata['valid']['images']    = [x[1] for x in szip['valid']]
	sdata['valid']['filenames'] = [x[2] for x in szip['valid']]	

	sdata['test']['labels']    = [x[0] for x in szip['test']]
	sdata['test']['images']    = [x[1] for x in szip['test']]
	sdata['test']['filenames'] = [x[2] for x in szip['test']]	

	print('train:{}, valid:{}, test:{}'.\
		format(len(sdata['train']['labels']), len(sdata['valid']['labels']), len(sdata['test']['labels'])))

	for key in sdata:
		sdata[key]['size'] = len(sdata[key]['labels'])

	return sdata	
"""