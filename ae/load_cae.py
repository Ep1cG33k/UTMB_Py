import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from model.custom_layers import DePool2D
from model.process import softMaxAxis
from math import floor


def init():
	with open('ae/versions/cae_v1.json', 'r') as f:
		loaded_model = tf.keras.models.model_from_json(f.read(), custom_objects={'DePool2D': DePool2D, 'softMaxAxis': softMaxAxis})
	loaded_model.load_weights("ae/versions/cae_v1.h5")
	print("Loaded CAE from disk")
	loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', 'mae'])
	graph = tf.get_default_graph()
	return loaded_model, graph