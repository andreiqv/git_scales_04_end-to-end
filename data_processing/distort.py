import tensorflow as tf
import math
#import sys


def augment_dataset(dataset, mult=1):
	#dataset.train_set = dataset.train_set.shuffle(60000).repeat(5).batch(128)
	#dataset = dataset.shuffle(60000).repeat(5).batch(16)
	#dataset = dataset.shuffle(3000).repeat(mult)
	dataset = dataset.repeat(mult)

	delta = {'hue':0.005, 'brightness':0.1, 'contrast':(0.8,1.2), 'saturation':(0.8,1.2)}

	def _random_distord(images, labels):

		images = tf.image.random_flip_left_right(images)
		images = tf.image.random_flip_up_down(images)

		d = 100
		#w, h = 299, 299
		w, h = 224, 224
		#paddings = tf.constant([[0, 0], [d, d], [d, d], [0, 0]])
		paddings = tf.constant([[d, d], [d, d], [0, 0]])
		#images = tf.pad(images, paddings, "CONSTANT")
		#images = tf.pad(images, paddings, "CONSTANT")
		images = tf.pad(images, paddings, "SYMMETRIC")
		#images = tf.image.resize_image_with_crop_or_pad(images, w+d, h+d)
		angle = tf.random_uniform(shape=(1,), minval=0, maxval=90)
		images = tf.contrib.image.rotate(images, angle * math.pi / 180, interpolation='BILINEAR')
		#images = tf.image.crop_to_bounding_box(images, d, d, s+d, s+d)
		images = tf.image.resize_image_with_crop_or_pad(images, w, h)			

		#images = tf.image.random_hue(images, max_delta=0.05)
		#images = tf.image.random_contrast(images, lower=0.3, upper=1.8)
		#images = tf.image.random_brightness(images, max_delta=0.3)
		#images = tf.image.random_saturation(images, lower=0.0, upper=2.0)

		#images = tf.image.random_hue(images, max_delta=0.02)
		#images = tf.image.random_contrast(images, lower=0.5, upper=1.5)
		#images = tf.image.random_brightness(images, max_delta=0.2)
		#images = tf.image.random_saturation(images, lower=0.5, upper=1.5)

		images = tf.image.random_hue(images, max_delta=delta['hue'])
		images = tf.image.random_brightness(images, max_delta=delta['brightness'])
		images = tf.image.random_contrast(images, lower=delta['contrast'][0], upper=delta['contrast'][1])
		images = tf.image.random_saturation(images, lower=delta['saturation'][0], upper=delta['saturation'][1])

		images = tf.minimum(images, 1.0)
		images = tf.maximum(images, 0.0)		

		return images, labels

	dataset = dataset.map(_random_distord)

	return dataset


def augment_dataset_no_labels(dataset, mult=1):
	#dataset.train_set = dataset.train_set.shuffle(60000).repeat(5).batch(128)
	#dataset = dataset.shuffle(60000).repeat(5).batch(16)
	dataset = dataset.shuffle(10000).repeat(mult)

	def _random_distord(images):

		images = tf.image.random_flip_left_right(images)
		#images = tf.image.random_hue(images, max_delta=0.05)
		#images = tf.image.random_contrast(images, lower=0.3, upper=1.8)
		#images = tf.image.random_brightness(images, max_delta=0.3)
		#images = tf.image.random_saturation(images, lower=0.0, upper=2.0)
		images = tf.image.random_hue(images, max_delta=0.02)
		images = tf.image.random_contrast(images, lower=0.5, upper=1.5)
		images = tf.image.random_brightness(images, max_delta=0.2)
		images = tf.image.random_saturation(images, lower=0.5, upper=1.5)
		images = tf.minimum(images, 255.0)
		images = tf.maximum(images, 0.0)
		return images

	dataset = dataset.map(_random_distord)

	return dataset




def augment_dataset_with_toss(dataset):
	#dataset.train_set = dataset.train_set.shuffle(60000).repeat(5).batch(128)
	#dataset = dataset.shuffle(60000).repeat(5).batch(16)
	dataset = dataset.shuffle(60000).repeat(5)

	def _random_distord(images, labels):
		rand = tf.random_uniform(shape=(1,), minval=0, maxval=2)
		toss = tf.less(rand, tf.constant([1.0], dtype=tf.float32))
		toss = tf.reshape(toss, [])

		def _flip_left():
			fliped_im = tf.image.flip_left_right(images)

			mirror = tf.constant([-1], dtype=tf.float32)
			mirror = tf.expand_dims(mirror, 0)
			mirror = tf.pad(mirror, [[0, 0], [0, tf.shape(labels)[1] - 1]], "CONSTANT", constant_values=1)
			mirror = tf.squeeze(mirror)
			mirror = tf.multiply(labels, mirror)  # [-0.34, 0.4, 0.28, 0,21, 1]

			basis = tf.constant([1], dtype=tf.float32)
			basis = tf.expand_dims(basis, 0)
			basis = tf.pad(basis, [[0, 0], [0, tf.shape(labels)[1] - 1]], "CONSTANT")  # [1, 0, 0 0 0]
			basis = tf.squeeze(basis)  # ?

			fliped_label = basis + mirror  # [0.6599, 0.4, 0.28, 0.21, 1]

			# eliminate fucking empty image coordinates problem
			# https://stackoverflow.com/questions/50538038/tf-data-dataset-mapmap-func-with-eager-mode
			new_labels = tf.multiply(fliped_label[:, 0], fliped_label[:, 4])
			new_labels = tf.expand_dims(new_labels, 1)
			new_labels = tf.concat([new_labels, fliped_label[:, 1:]], 1)

			return fliped_im, new_labels

		images, labels = tf.cond(toss, _flip_left, lambda: (images, labels))

		images = tf.image.random_hue(images, max_delta=0.05)
		images = tf.image.random_contrast(images, lower=0.3, upper=1.8)
		images = tf.image.random_brightness(images, max_delta=0.3)
		images = tf.image.random_saturation(images, lower=0.0, upper=2.0)

		images = tf.minimum(images, 1.0)
		images = tf.maximum(images, 0.0)

		return images, labels

	dataset = dataset.map(_random_distord)

	return dataset


augment_dataset_2 = augment_dataset