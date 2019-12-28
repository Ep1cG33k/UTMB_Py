import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.activations import softmax

def tversky_loss(y_true, y_pred):
	alpha = 0.5
	beta = 0.5

	ones = K.ones(K.shape(y_true))
	p0 = y_pred  # probability that voxels are class i; voxel is pixel in 3D
	p1 = ones - y_pred  # probability that voxels are not class i
	g0 = y_true
	g1 = ones - y_true

	num = K.sum(p0 * g0, (0, 1, 2, 3))
	den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

	T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

	Ncl = K.cast(K.shape(y_true)[-1], 'float32')
	return Ncl - T


def dice_loss(y_true, y_pred):
	alpha = 0.5
	beta = 0.5

	ones = K.ones(K.shape(y_true))
	p0 = y_pred  # probability that voxels are class i; voxel is pixel in 3D
	p1 = ones - y_pred  # probability that voxels are not class i
	g0 = y_true
	g1 = ones - y_true

	num = K.sum(p0 * g0, (0, 1, 2, 3))
	den = num + alpha * K.sum(K.pow(p0, 2), (0, 1, 2, 3)) + beta * K.sum(K.pow(g0, 2), (0, 1, 2, 3))

	T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

	# Ncl = K.cast(K.shape(y_true)[-1], 'float32')
	return 1 - T


def focal_loss(y_true, y_pred, gamma=2):
	y_pred /= K.sum(y_true, axis=-1, keepdims=True)
	eps = K.epsilon()
	y_pred = K.clip(y_pred, eps, 1. - eps)
	return -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred), axis=-1)


# vectorize this function?
def pixel_weight(pixel, omega1, omega2):
	w1 = omega1 * (1 if np.argmax(pixel, axis=-1) == 1 else 0)
	w2 = omega2 * (0 if np.argmax(pixel, axis=-1) == 0 or np.argmax(pixel, axis=-1) == 9 else 1)
	return 1 + w1 + w2


weights = np.vectorize(pixel_weight)


def weighted_multi_class_logistic_loss(y_true, y_pred):
	return -K.sum(weights(y_pred, 10, 5) * y_true * K.log(y_pred), axis=-1)


def full_loss(y_true, y_pred, lambda1=0.5, lambda2=1):
	l1 = lambda1 * dice_loss(y_true, y_pred)
	l2 = lambda2 * weighted_multi_class_logistic_loss(y_true, y_pred)
	return l1 + l2


def dice_coef(y_true, y_pred, smooth=1):
	intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
	union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])

	return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def mean_iou(y_true, y_pred):
	prec = []
	for t in np.arange(0.5, 1.0, 0.05):
		y_pred_ = tf.to_int32(y_pred > t)
		score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
	return K.mean(K.stack(prec), axis=0)


def softMaxAxis(x):
	return softmax(x, axis=-1)