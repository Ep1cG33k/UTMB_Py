import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from math import floor


def init():
	# json_file = open('model/model.json','r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	
	with open('ae/versions/vae_v2.json', 'r') as f:
		loaded_model = tf.keras.models.model_from_json(f.read())
	# load woeights into new model
	loaded_model.load_weights("ae/versions/vae_v2.h5")
	print("Loaded AE from disk")
	
	# compile and evaluate loaded model
	
	# loaded_model = tf.keras.models.load_model('model/best_model.h5', custom_objects={'DePool2D': DePool2D, 'softMaxAxis': softMaxAxis})
	loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae', 'accuracy'])
	# loss,accuracy = model.evaluate(X_test,y_test)
	# print('loss:', loss)
	# print('accuracy:', accuracy)
	graph = tf.get_default_graph()
	
	return loaded_model, graph