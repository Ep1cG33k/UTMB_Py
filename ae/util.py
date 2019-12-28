import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from math import floor

int_to_layer = {
    0 : 'rnfl',
    1 : 'gcl-ipl',
    2 : 'inl',
    3 : 'opl',
    4 : 'onl-ism',
    5 : 'ise',
    6 : 'os-rpe'
}

def normalize(data, param_data=None):
	if param_data is None:
		minx = 0
		maxx = 1.65359375e-4
	else:
		minx = np.amin(param_data)
		maxx = np.amax(param_data)
		
	data = (data-minx)/(maxx-minx)
	return data


def denormalize(data, param_data=None):
	if param_data is None:
		minx = 0
		maxx = 1.65359375e-4
	else:
		minx = np.amin(param_data)
		maxx = np.amax(param_data)
	
	data = ((maxx - minx) * data) + minx
	return data

def reconstruction_loss(x, p):
	rl = np.square((x-p))
	#rl = (x - p) * (x - p)
	return rl

def n_max_indices(data, n=10):
	test_rl_np_c = np.copy(data)
	# dictionary of indices and max reconstruction losses
	test_max = dict()
	for i in range(n):
		test_max_rl = np.unravel_index(np.argmax(test_rl_np_c), test_rl_np_c.shape)
		# test_max.append(test_max_rl)
		test_max.update({test_max_rl: test_rl_np_c[test_max_rl]})
		test_rl_np_c[test_max_rl] = 0.0
	return test_max


def compare(x, p, indices, pandas=True):
	if pandas == True:
		x = pd.DataFrame.to_numpy(x)
		p = pd.DataFrame.to_numpy(p)
	comp = dict()
	for key in indices:
		comp.update({x[key]: p[key]})
	return comp


def index_to_loc(indices, twod=False):
	locations = []
	if twod:
		for key in indices:
			loc = [floor(key[0] / 8), key[0] % 8, int_to_layer.get(key[1]), indices.get(key), key[1], key[2]]
			locations.append(loc)
	else:
		
		for key in indices:
			loc = [floor(key[0] / 8), key[0] % 8, int_to_layer.get(key[1]), indices.get(key)]
			locations.append(loc)
	return np.array(locations)

def anomalies(x, p, n=10, twod=False):
	rl = reconstruction_loss(x, p)
	idx = n_max_indices(rl, n)
	#comparison = compare(x, p, idx, pandas=True)
	loc = index_to_loc(idx, twod=twod)
	return rl, loc