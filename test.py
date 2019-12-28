import numpy as np
import scipy
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import io
import sys
from tifffile import imwrite
import os
from os import listdir
from tqdm import tqdm
from os.path import isfile, join
from ae import load_ae, util, load_cae
import anomaly

# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from model import load, process

# global vars for easy reusability
global model, graph, ae, ae_graph, cae, cae_graph
# initialize these variables
model, graph = load.init()
ae, ae_graph = load_ae.init()
cae, cae_graph = load_cae.init()

def fill_dict(d):
	'''
		Auxiliary function to fill dictionary with key values from 2 to 9.

		:param dict d: Dictionary to be filled
		:return dict d: Filled dictionary

	'''
	for i in range(2,9):
		if i not in d:
			d[i] = 0
	return d

def integrate(tensor, num):
	'''
		Count number of pixels for each layer and store in dictionary.

		:param np tensor: Segmented numpy array that needs to be counted
		:param int num: Number of layer whose frequency is returned (i.e. 2 -> rnfl pixels)
		:return float freq[num]: Frequency of specified layer in # of pixels
	'''
	unique, counts = np.unique(tensor, return_counts=True)
	freq = dict(zip(unique, counts))
	freq = fill_dict(freq)
	return freq[num]

def convert_to_microns(n):
	'''
		Convert number of pixels to micron^3 (volume)

		:param int n: Number of pixels
		:return float microns_cubed: Volume in microns cubed
	'''
	microns_cubed = n * 1.9e-6
	return microns_cubed

def thickness(img):
	rnfl = convert_to_microns((integrate(img, 2) / img.shape[-1])/img.shape[0])
	ipl = convert_to_microns((integrate(img, 3) / img.shape[-1])/img.shape[0])
	inl = convert_to_microns((integrate(img, 4) / img.shape[-1])/img.shape[0])
	opl = convert_to_microns((integrate(img, 5) / img.shape[-1])/img.shape[0])
	is_onl = convert_to_microns((integrate(img, 6) / img.shape[-1])/img.shape[0])
	os = convert_to_microns((integrate(img, 7) / img.shape[-1])/img.shape[0])
	rpe = convert_to_microns((integrate(img, 8) / img.shape[-1])/img.shape[0])
	thicc = [rnfl, ipl, inl, opl, is_onl, os, rpe]
	return thicc

def loc_thickness(img, slices):
	rnfl = convert_to_microns(integrate(img, 2)/(512/slices))
	ipl = convert_to_microns(integrate(img, 3)/(512/slices))
	inl = convert_to_microns(integrate(img, 4)/(512/slices))
	opl = convert_to_microns(integrate(img, 5)/(512/slices))
	is_onl = convert_to_microns(integrate(img, 6)/(512/slices))
	os = convert_to_microns(integrate(img, 7)/(512/slices))
	rpe = convert_to_microns(integrate(img, 8)/(512/slices))
	thicc = [rnfl, ipl, inl, opl, is_onl, os, rpe]
	return thicc

def variance(img):
	local_thick = np.apply_along_axis(loc_thickness, 1, img, 512)
	std = np.std(local_thick, axis=-1)
	cov = scipy.stats.variation(local_thick, axis=-1)
	return std, cov

def local_avg(img, slicesz, slicesx, perimage=True):
	img = np.expand_dims(img, -1)
	img_arr = process.slice(img, slicesz, slicesx, concat=False)
	grid = []
	for image in img_arr:
		x = []
		for strip in image:
			x.append(loc_thickness(strip, img_arr.shape[1]))
		grid.append(np.transpose(x))
	#local_thick = np.apply_along_axis(loc_thickness, 1, img_arr, slicesx)
	grid = np.array(grid)
	
	#avg along slice axis
	avg_grid = np.mean(grid, axis=0)
	return grid if perimage else avg_grid


def edema_vol(img):
	def edema_integrate(tensor):
		unique, counts = np.unique(tensor, return_counts=True)
		freq = dict(zip(unique, counts))
		freq = fill_dict(freq)
		return freq[1]
	
	def convert_to_microns_vol(n):
		microns_cubed = n * 5393.1933e-15
		return microns_cubed
	
	return convert_to_microns_vol(edema_integrate(img[range(0, img.shape[0])]))

def predict(file_path, anomaly_detection='none', slices=1000, anomalies=50):
	'''
		Given file_path, return segmented prediction as numpy array

		:param string file_path: File path as string of 256x512 image to be segmented
		:param bool anomaly_detection: True for anomaly detection indices saved as csv, False if not
		:param int slices: Number of slices in OCT B-Scan/Axial (i.e. 1000), optional parameter, default=1000
		:return np twod: 3D numpy array of segmented image

	'''
	
	# Read file from file_path
	x = io.imread(file_path)
	# Image pre-processing
	x = gaussian_filter(x, sigma=2)
	#x = maximum_filter(x, size=(1, 2, 7))
	x = np.expand_dims(x, axis=-1)
	x = x / 255
	# Slice image into 256x64
	x = process.slice(x, slices)
	
	if anomaly_detection == 'oct':
		# Uses CONVOLUTIONAL AUTOENCODER: Determines anomalies based directly on OCT scans, NOT predicted thickness
		with cae_graph.as_default():
			# perform the anomaly detection prediction
			p = cae.predict(x)
			# Save reconstruction loss and indices of anomalies as numpy arrays
			rl, loc = util.anomalies(x, p, n=anomalies, twod=True)
			# Save anomalies into csv
			if not os.path.exists('predictions/{}/anomalies'.format(file_path)):
				os.makedirs('predictions/{}/anomalies'.format(file_path))
			np.savetxt('predictions/{}/anomalies/cae_anomalies.csv'.format(file_path), X=loc, delimiter=',', fmt='%s')
	
	with graph.as_default():
		# Perform the prediction
		out = model.predict(x)
		# Concatenate back to 256x512
		y = process.concat(out, slices * 8)
		# Convert three dimensional arrary with one hot vectors to two dimensions
		twod = process.threed_one_hot_to_twod(y)
		twod = np.array(twod, dtype='uint8')
		return twod
	
def full_predict(file_path, anomaly_detection='none', slicesx=8, slices=1000, anomalies=50):
	'''
		Complete prediction function that segments, calculates quantitative data, and saves to predictions file
		:param string file_path: File of image to be segmented
		:param bool anomaly_detection: True for anomaly detection indices saved as csv, False if not
		:param int slicesx: Number of slices in x direction, default=8
		:param int slices: Number of slices in OCT B-Scan/Axial, default=1000
		:return np twod: 3D numpy array of segmented image
	'''
	
	#If directory does not yet exist, create one for file
	if not os.path.exists('predictions/{}'.format(file_path)):
		os.makedirs('predictions/{}'.format(file_path))
		print("Directory ", 'predictions/{}'.format(file_path), " Created ")
	else:
		print("Directory ", 'predictions/{}'.format(file_path), " already exists")
		
	#Segmentation
	twod = predict(file_path, anomaly_detection=anomaly_detection, slices=slices, anomalies=anomalies)
	imwrite('predictions/{}/segment.tif'.format(file_path), twod * 28)

	#Average Thickness
	avg = np.array(thickness(twod))
	np.savetxt('predictions/{}/avg.csv'.format(file_path), X=avg, delimiter=',')
	
	#Standard Deviation
	std = variance(twod)[0]
	np.savetxt('predictions/{}/std.csv'.format(file_path), X=std, delimiter=',')
	
	#Coefficient of Variance
	cov = variance(twod)[1]
	np.savetxt('predictions/{}/cov.csv'.format(file_path), X=cov, delimiter=',', fmt='%-7.4f')
	
	#Local Averages per Strip per Slice
	loc_avg_per = local_avg(twod, twod.shape[0], slicesx, perimage=True)
	loc_avg_csv = np.transpose(loc_avg_per, (0,2,1))
	loc_avg_concat = np.concatenate([loc_avg_csv[i] for i in range(0, loc_avg_csv.shape[0])], axis=0)
	np.savetxt('predictions/{}/local_avg_per.csv'.format(file_path), X=loc_avg_concat, delimiter=',')

	# Write the array to disk

	with open('predictions/{}/local_avg_per_slice.txt'.format(file_path), 'w') as outfile:
		# I'm writing a header here just for the sake of readability
		# Any line starting with "#" will be ignored by numpy.loadtxt
		outfile.write('# Array shape: {0}\n'.format(loc_avg_per.shape))
		for data_slice in loc_avg_per:
			np.savetxt(outfile, data_slice)
			
			# Writing out a break to indicate different slices...
			outfile.write('# New slice\n')
	
	#Local Averages per Slice
	local_avg_avg = local_avg(twod, twod.shape[0], slicesx, perimage=False)
	np.savetxt('predictions/{}/local_avg.csv'.format(file_path), X=local_avg_avg, delimiter=',')
	
	# Anomaly with predicted layer thickness
	if anomaly_detection == 'csv':
		if not os.path.exists('predictions/{}/anomalies'.format(file_path)):
			os.makedirs('predictions/{}/anomalies'.format(file_path))
		anomaly.predict_from_thickness('predictions/{}/local_avg_per.csv'.format(file_path), f_file_path='{}'.format(file_path), anomalies=anomalies)
		
	# Edema Volume Calculation
	edema = edema_vol(twod).astype('str')
	with open('predictions/{}/edema_vol.txt'.format(file_path), 'w') as outfile:
		outfile.write(edema)
		
	return twod

# Code that is run
if __name__ == "__main__":
	
	# Make list of all files in ../data/ to iterate over
	onlyfiles = [f for f in listdir('data/oct') if isfile(join('data/oct', f))]
	onlyfiles = ['data/oct/' + s for s in onlyfiles]
	print(onlyfiles)
	
	#For each file in directory ../data/ perform segmentation and relevant calculations
	for f in tqdm(onlyfiles):
		full_predict(f, anomaly_detection='none', slices=100, anomalies=50)
