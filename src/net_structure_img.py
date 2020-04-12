import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io

def img_net_strucuture(input_image, input_dim,HIDDEN_DIM,bit):
	net = {}
	ops = []
	current =tf.convert_to_tensor(input_image,dtype='float32')


	Wq_1 = tf.random_normal([input_dim,HIDDEN_DIM],stddev=1.0)*0.01
	Wq_2 = tf.random_normal([HIDDEN_DIM,bit],stddev=1.0)*0.01
	Bq_1 = tf.random_normal([HIDDEN_DIM],stddev=1.0)*0.01
	Bq_2 = tf.random_normal([bit],stddev=1.0)*0.01

	w1 = tf.Variable(Wq_1, name='w' + str(1))
	b1 = tf.Variable(Bq_1, name='bias' + str(1))
	w2 = tf.Variable(Wq_2, name='w' + str(2))
	b2 = tf.Variable(Bq_2, name='bias' + str(2))

	# fc8 = tf.nn.xw_plus_b(
	# 	tf.nn.xw_plus_b(current, w1, b1), w2,b2)

	fc8 = tf.nn.xw_plus_b(
		tf.nn.tanh(tf.nn.xw_plus_b(current, w1, b1)), w2,b2)

	ops.append(w1)
	ops.append(b1)
	ops.append(w2)
	ops.append(b2)

	net['fc8'] = fc8

	return net

