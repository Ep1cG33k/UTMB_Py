import numpy as np
import tqdm
from keras.activations import softmax

subject_path = ['Subject_0{}.mat'.format(i) for i in range(1, 10)] + ['Subject_10.mat']
data_indices = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
data_index_matrix = [
    data_indices,
    data_indices,
    data_indices,
    [i - 4 for i in data_indices],
    [i - 2 for i in data_indices],
    data_indices,
    [i + 1 for i in data_indices],
    [i - 2 for i in data_indices],
    data_indices,
    data_indices
]

features = ['images', 'manualFluid1', 'manualFluid2', 'manualLayers1', 'manualLayers2', 'automaticFluidDME', 'automaticLayersDME', 'automaticLayersNormal']

def transpose_t(img, fluid_1, fluid_2, layers, indices=(2, 0, 1)):
	'''
	 Transpose each tensor 2, 0, 1, meaning swap dimensions from 0,1,2 to 2,0,1, or diff, depending on indices param

	 :param np img: Numpy img tensor raw from .mat 496x768x61
	 :param np fluid_1: Numpy fluid1 tensor raw from .mat 496x768x61
	 :param np fluid_2: Numpy fluid2 tensor raw from .mat 496x768x61
	 :param np layers: Numpy layers tensor raw from .mat 8x768x61
	 :param np tuple: indices tuple for transposition
	 :return np img_array: Transposed img tensor 61x496x768
	 :return np manual_fluid_array_1: Transposed fluid1 tensor 61x496x768
	 :return np manual_fluid_array_2: Transposed fluid2 tensor 61x496x768
	 :return np manual_layers_array: Transposed layers tensor 61x8x768

	 '''

	img_array = np.transpose(img, indices)
	manual_fluid_array_1 = np.transpose(fluid_1, indices)
	manual_fluid_array_2 = np.transpose(fluid_2, indices)
	manual_layers_array = np.transpose(layers, indices)

	return img_array, manual_fluid_array_1, manual_fluid_array_2, manual_layers_array


def extract(subject, mats, transpose=True):
	'''
	Extract each relevant feature from the subject, i.e. image, fluid, layers

	:param int subject: The mat subject, 0-9
	:param list mats: List of mats, raw or cropped
	:return np img_array: Transposed img tensor 61x496x768
	:return np manual_fluid_array_1: Transposed fluid1 tensor 61x496x768
	:return np manual_fluid_array_2: Transposed fluid2 tensor 61x496x768
	:return np manual_layers_array: Transposed layers tensor 61x8x768

	'''

	img_array = mats[subject]['images']
	manual_fluid_array_1 = mats[subject]['manualFluid1']
	manual_fluid_array_2 = mats[subject]['manualFluid2']
	manual_layers_array = mats[subject]['manualLayers1']

	if transpose:
		img_array, manual_fluid_array_1, manual_fluid_array_2, manual_layers_array = transpose_t(
			img_array, manual_fluid_array_1, manual_fluid_array_2, manual_layers_array)

	return img_array, manual_fluid_array_1, manual_fluid_array_2, manual_layers_array


def crop(subject, mat, bounds_vertical, bounds_horizontal, labeled_only=False):
	'''
	 Crop one cross-section to remove excess space

	 :param int subject number: The mat subject, 0-9
	 :param tuple bounds_vertical: Vertical crop boundary
	 :param tuple bounds_horizontal: Horizontal crop boundary
	 :param bool labeled_only: If crop_sub should only contains labeled b-scan sections
	 :return list crop_sub: Return list of img, two fluids, and layers cropped

	 '''

	img_array, manual_fluid_array_1, manual_fluid_array_2, manual_layers_array = extract(subject, mat)

	crop_img_stack = []
	crop_fluid1_stack = []
	crop_fluid2_stack = []
	crop_layer_stack = []

	for i in range(0, 61) if not labeled_only else data_index_matrix[subject]:
		cropped_img = img_array[i][bounds_vertical[0]:bounds_vertical[1], bounds_horizontal[0]:bounds_horizontal[1]]
		cropped_fluid_1 = manual_fluid_array_1[i][bounds_vertical[0]:bounds_vertical[1],
						  bounds_horizontal[0]:bounds_horizontal[1]]
		cropped_fluid_2 = manual_fluid_array_2[i][bounds_vertical[0]:bounds_vertical[1],
						  bounds_horizontal[0]:bounds_horizontal[1]]
		cropped_layers = manual_layers_array[i][:, bounds_horizontal[0]:bounds_horizontal[1]] - (bounds_vertical[0])

		crop_img_stack.append(cropped_img)
		crop_fluid1_stack.append(cropped_fluid_1)
		crop_fluid2_stack.append(cropped_fluid_2)
		crop_layer_stack.append(cropped_layers)

	crop_img_stack = np.asarray(crop_img_stack)
	crop_fluid1_stack = np.asarray(crop_fluid1_stack)
	crop_fluid2_stack = np.asarray(crop_fluid2_stack)
	crop_layer_stack = np.asarray(crop_layer_stack)

	# TODO: crop_sub should be dict!
	crop_sub = {
		'images': crop_img_stack,
		'manualFluid1': crop_fluid1_stack,
		'manualFluid2': crop_fluid2_stack,
		'manualLayers1': crop_layer_stack
	}

	return crop_sub

#DATA PROCESSING FUNCTIONS:

def thresh(x):
	if x == 0:
		return 0
	else:
		return 1


# function is vectorized
thresh = np.vectorize(thresh, otypes=[np.float])


def normalize(img_array):
	img_array = img_array / 255
	return img_array


def twod_to_one_hot_threed(a):
	threed_one_hot = (np.arange(a.max() + 1) == a[..., None]).astype(int)
	return threed_one_hot


def threed_one_hot_to_twod(labels_one_hot):
	'''
	Convert 3D one hot tensor into 2d matrix with class values

	:param np labels_one_hot: The 4D tensor with 3D one-hot tensors, nx256x512x10
	:return np y: 3D nd array with 2D matrices with classes labeled, nx256x512

	'''
	
	return  np.argmax(labels_one_hot, axis=-1)

def create_dataset(sub_range):
	'''
	 Create a dataset from data range given with x and y

	 :param: tuple sub_range: The mat subject range, i.e. (0,5) subjects 0 to 4 will be used
	 :return np x: Return numpy array of x values
	 :return np y: Return numpy array of y values

	 '''
	x = []
	for sub in tqdm(range(sub_range[0], sub_range[1])):
		img_array, manual_fluid_array_1, manual_fluid_array_2, manual_layers_array = extract(sub, mats,	transpose=False)
		# Normalize image data between 0 and 1
		img_array = normalize(img_array)
		# manual_fluid_array_1 = thresh(manual_fluid_array_1)
		manual_fluid_array_1 = np.nan_to_num(manual_fluid_array_1)
		manual_fluid_array_1[manual_fluid_array_1 > 0] = 1
		print(sub)
		for index in range(0, 11):
			if sub == 6:
				for row in manual_layers_array[index]:
					row[range(0, 6)] = row[6]
					row[range(506, 512)] = row[506]
			elif sub == 8:
				for row in manual_layers_array[index]:
					row[range(500, 512)] = row[500]
			# return nd array
			x.append(img_array[index])

	'''
	#tqdm is progress bar
	for path in tqdm(paths):
	  mats
	  img_tensor = mat['images']
	  fluid_tensor = mat['manualFluid1']
  
	  img_array = np.transpose(img_tensor, (2, 0 ,1)) / 255
	  img_array = resize(img_array, (img_array.shape[0], width, height))
	  fluid_array = np.transpose(fluid_tensor, (2, 0 ,1))
	  #does the computation for each element in fluid_array (turn to 0 or 1 for normalization) 
	  fluid_array = thresh(fluid_array)
	  fluid_array  = resize(fluid_array, (fluid_array .shape[0], width_out, height_out))
  
	  for idx in data_indexes:
		x += [np.expand_dims(img_array[idx], 0)]
		y += [np.expand_dims(fluid_array[idx], 0)]
	'''

	return np.expand_dims(np.array(x), axis=3)

def slice(x, slicesz, slicesx=8, concat=True):
  x = np.split(x, slicesx, axis=2)
  x = np.transpose(x, (1,0,2,3,4))
  x_ar = np.concatenate([x[i] for i in range(0, slicesz)], axis=0)
  return x_ar if concat==True else x


def blend(strip1, strip2, div=16, overlap=2):
	strip1 = np.split(strip1, div, axis=1)
	strip2 = np.split(strip2, div, axis=1)
	for i in range(0, overlap):
		# blending avg.
		new_strip = (strip1[-(i + 1)] + strip2[overlap - (i + 1)]) / 2
		strip1[-(i + 1)] = new_strip
	
	strip1 = np.concatenate([strip1[i] for i in range(0, div)], axis=1)
	return strip1

def concat(y_pred, slices):
	y_pred_concat = []
	i = 0
	while i < slices:
		mix = np.concatenate([blend(y_pred[j], y_pred[j+1]) if j % 8 != 7 else y_pred[j] for j in range(i, i + 8)], axis=1)
		y_pred_concat.append(mix)
		i = i + 8
	
	return np.array(y_pred_concat)

def softMaxAxis(x):
    return softmax(x,axis=-1)