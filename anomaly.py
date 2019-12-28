import numpy as np
from numpy import genfromtxt
import scipy
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import io
import sys
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
# import io
import test
# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from model import process
from ae import load_ae, util, load_cae

ae, ae_graph = load_ae.init()
cae, cae_graph = load_cae.init()

def detect_anomalies(x, top_n=50, return_prediction=False):
	with ae_graph.as_default():
		x = util.normalize(x)
		p = ae.predict(x)
		#x = util.denormalize(x)
		rl = util.reconstruction_loss(x, p)
		p = util.denormalize(p)
		max_idx = util.n_max_indices(rl, top_n)
		location = util.index_to_loc(max_idx)
		#returns numpy array of list containing slice, strip #s, layer and rl
		if return_prediction == False:
			return location
		else:
			return location, p, rl
		
#File used for ONLY anomaly detection using either variational autoencoder (from thickness data) or convolutional autoencoder (from OCT scans)
def predict_from_2D(file_path, slices):
	x = io.imread(file_path)
	x = gaussian_filter(x, sigma=2)
	x = maximum_filter(x, size=(1, 2, 7))
	x = np.expand_dims(x, axis=-1)
	x = x / 255
	x = process.slice(x, slices)
	
	with cae_graph.as_default():
		# perform the anomaly detection prediction magic
		p = cae.predict(x)
		rl, loc = util.anomalies(x, p, n=50, twod=True)
		np.savetxt('predictions/{}/anomalies/anom.csv'.format(file_path), X=loc, delimiter=',', fmt='%s')
		
def predict_from_thickness(file_path, save_p=False, save_rl=False, f_file_path='', anomalies=50):
	x = genfromtxt(file_path, delimiter=',')
	anomalies, p, rl = detect_anomalies(x, top_n=anomalies, return_prediction=True)
	np.savetxt('predictions/{}/anomalies/anom.csv'.format(f_file_path if f_file_path != '' else file_path), X=anomalies, delimiter=',', fmt='%s')
	
	if save_p is True: np.savetxt('predictions/{}/anomalies/p.csv', X=p, delimiter=',')
	if save_rl is True: np.savetxt('predictions/{}/anomalies/lr.csv', X=rl, delimiter=',')
	
def predict_anomaly(slices=1000, cae=True):
	if cae is True:
		octfiles = [f for f in listdir('data/oct') if isfile(join('data/oct', f))]
		octfiles = ['data/oct/' + s for s in octfiles]
		for f in tqdm(octfiles):
			predict_from_2D(f, slices=slices)
	else:
		csvfiles = [f for f in listdir('data/csv') if isfile(join('data/csv', f))]
		csvfiles = ['data/csv/' + s for s in csvfiles]
		for f in tqdm(csvfiles):
			predict_from_thickness(f)

if __name__ == "__main__":
	predict_anomaly()
	
	
		
