
from flask import Flask, render_template, request, send_file, Response
import numpy as np
import scipy
from scipy import io as scipyio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import keras.models
import re
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter

from skimage import io
import sys
from tifffile import imagej_description_metadata, imread, imwrite
import os
from tempfile import TemporaryFile
#import io

# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from model import load, process

# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph
# initialize these variables
model, graph = load.init()


# decoding an image from base64 into raw representation

@app.route('/')
def index():
	# initModel()
	# render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/test', methods=['GET'])
def test():
	return 'Bruh'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	
	print("here!!!")
	#TODO: Make program compatible with .tif file, direct
	file = request.files['image-file']
	#maybe do something with BytesIO
	print('here2')
	#TIFF FILES
	#tif = np.array(Image.open(file))
	#tif = plt.imread(file)
	#tif = imread(file)
	
	file_path = 'tif/cropped_test.tif'
	tif = io.imread(file_path)
	print(tif.shape)
	x = io.imread(file)
	print(x.shape)
	# read the image into memory
	x = gaussian_filter(x, sigma=2)
	x = maximum_filter(x, size=(2, 7))
	x = np.expand_dims(x, axis=-1)
	
	x = np.expand_dims(x, axis=0)
	
	
	#MAT FILES
	'''
	 mat = scipyio.loadmat(file)
	images = mat['images']
	images = np.transpose(images, (2, 0, 1))
	x = images[18]
	print(x.shape)
	x = x[75:331, 99:611]
	x = np.expand_dims(x, axis=-1)
	x = np.expand_dims(x, axis=0)
	'''
	x = x/255
	x = process.slice(x, 1)
	# convert to a 4D tensor to feed into our model
	print("debug2")
	# in our computation graph
	with graph.as_default():
		# perform the prediction
		out = model.predict(x)
		y = process.concat(out, 8)
		twod = process.threed_one_hot_to_twod(y)
		print(twod.shape)
		plt.imshow(twod.reshape((256, 512)))
		plt.colorbar()
		plt.show()
		np.save('numpy.npy', twod)
		imwrite('segment.tif', twod)
		print("debug3")
		# convert the response to a string
		#response = np.array_str(np.argmax(out, axis=1))
		fig = create_figure(twod, 0)
		#output = io.BytesIO()
		#FigureCanvas(fig).print_png(output)
		#return Response(output.getvalue(), mimetype='image/png')
		return "Finished"

def create_figure(numpy, index):
	fig = Figure()
	axis = fig.add_subplot(1,1,1)
	axis.imshow(numpy[index])
	#img_plot = plt.imshow(numpy)
	return fig

@app.route("/upload-image")
def upload_image():
	if request.method == "POST":
		
		if request.files:
			image = request.files["image"]
			
			print(image)
'''
def return_files_tut():
    try:
        return send_file()
    except Exception as e:
        return str(e)

'''

@app.route('/return-files')
def return_files_tut():
	try:
		return send_file('segment.tif')
	except Exception as e:
		return str(e)


if __name__ == "__main__":
	# decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	# run the app locally on the givn port
	app.run(host='127.0.0.1', port=port)
	# optional if we want to run in debugging mode
	app.run(debug=True)