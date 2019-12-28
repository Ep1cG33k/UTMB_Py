import numpy as np
import tensorflow as tf
import tensorflow.keras.layers
from keras.models import model_from_json, load_model
from keras.layers import UpSampling2D
from model.loss import full_loss, dice_coef
from model.custom_layers import DePool2D
from model.process import softMaxAxis

def init():

	#json_file = open('model/model.json','r')
	#loaded_model_json = json_file.read()
	#json_file.close()
	#loaded_model = model_from_json(loaded_model_json)
	
	
	with open('model/versions/model_v26.json', 'r') as f:
		loaded_model = tf.keras.models.model_from_json(f.read() , custom_objects={'DePool2D': DePool2D, 'softMaxAxis': softMaxAxis})
	#load woeights into new model
	loaded_model.load_weights("model/versions/model_v26.h5")
	print("Loaded Model from disk")
	
	

	#compile and evaluate loaded model
	
	
	#loaded_model = tf.keras.models.load_model('model/best_model.h5', custom_objects={'DePool2D': DePool2D, 'softMaxAxis': softMaxAxis})
	loaded_model.compile(loss=[full_loss],optimizer='adam',metrics=[dice_coef])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph