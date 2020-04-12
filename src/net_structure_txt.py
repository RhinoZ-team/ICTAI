import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io
LAYER1_NODE = 512

def txt_net_strucuture(text_input, input_dim, bit):

	W_fc8 = tf.random_normal([1, input_dim, 1, LAYER1_NODE], stddev=1.0) * 0.01
	b_fc8 = tf.random_normal([1, LAYER1_NODE], stddev=1.0) * 0.01
	fc1W = tf.Variable(W_fc8)
	fc1b = tf.Variable(b_fc8)

	conv1 = tf.nn.conv2d(text_input, fc1W, strides=[1, 1, 1, 1], padding='VALID')
	layer1 = tf.nn.relu(tf.nn.bias_add(conv1, tf.squeeze(fc1b)))

	W_fc2 = tf.random_normal([1, 1, LAYER1_NODE, bit], stddev=1.0) * 0.01
	b_fc2 = tf.random_normal([1, bit], stddev=1.0) * 0.01
	fc2W = tf.Variable(W_fc2)
	fc2b = tf.Variable(b_fc2)

	conv2 = tf.nn.conv2d(layer1, fc2W, strides=[1, 1, 1, 1], padding='VALID')
	output_g = tf.squeeze(tf.nn.bias_add(conv2, tf.squeeze(fc2b)))

	# current = tf.convert_to_tensor(text_input, dtype='float32')
	#
	# Wq_1 = tf.random_normal([input_dim, bit], stddev=1.0) * 0.01
	# Bq_1 = tf.random_normal([bit], stddev=1.0) * 0.01
	#
	# w1 = tf.Variable(Wq_1, name='w' + str(1))
	# b1 = tf.Variable(Bq_1, name='bias' + str(1))
	#
	# output_g = tf.nn.xw_plus_b(current, w1, b1)

	return output_g
