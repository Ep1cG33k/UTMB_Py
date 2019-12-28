#ReLSPAD
######Updated as of July 29, 2019
ReLSPAD is a Retinal Layer Segmentation, Pathology Quantification, and Anomaly Detection tool that uses Deep Learning.

![alt text][fig2]

[fig2]: static/segmentation.png "Figure 2: Segmentation"

![alt text][fig3]

[fig3]: static/seg2.png "Figure 3: Anomaly Detection and Application"
##Purpose

The diagnosis of retinal diseases can be accelerated by the non-invasive detection and quantification of pathology yielding the examination of the morphological features of retinal layers. Recent developments in the field of deep learning have led to improvements in quantitative visualization of  medical images. Therefore, a deep learning model is designed and implemented to segment retinal layers more accurately than a human expert and outperform traditional quantification modalities in cost, efficiency, accuracy, and volume. In addition to the segmented image, local averages, fluid volume, and variance metrics are calculated by the algorithm to facilitate data analysis and diagnosis. Furthermore, a variational autoencoder is employed to detect and visualize statistical anomalies in the segmented data. The segmentation results with the greatest performance are overlaid with vessel-enhanced scans to visualize vascular beds and pathology features from an orthographic perspective. As a result, retinal layers and pathology may be analyzed more precisely and more efficiently compared to prior methods, thereby advancing the diagnosis of retinal diseases. 

##Usage

###Segmentation and Quantification
1) Upload 256x512xn OCT B-Scan tif files to **UTMB_Py/data/oct folder** (will act as queue for segmentation)
2) Open **UTMB_Py/test.py** in Pycharm. Make sure Python Interpreter is set to Python 3.7 with all used libraries downloaded. 
3) In **test.py**, see: 
```python 
if __name__ == "__main__":
	
	# Make list of all files in ../data/ to iterate over
	onlyfiles = [f for f in listdir('data') if isfile(join('data', f))]
	onlyfiles = ['data/' + s for s in onlyfiles]
	
	#For each file in directory ../data/ perform segmentation and relevant calculations
	for f in tqdm(onlyfiles):
		full_predict(f, anomaly_detection=False, slices=1000, anomalies=50)
```
4) Update parameters of _full_predict_ function accordingly. Slices must equal the y-dimension in data (i.e. **n** in 256x512x**n**) for full segmentation. If slices=n, then the algorithm will only segment the first n slices.  

**Anomaly Detection Parameters:**
 
 _anomaly_detection='oct'_ : Anomalies are detected directly from OCT images. Slightly longer.
 
 _anomaly_detection='csv'_ : Anomalies are detected using predicted local averages based on segmentation.
 
 _anomaly_detection='none'_ : No anomaly indices are returned.


You may change the number of anomalies to detect (top-n) but adjusting anomalies parameter. Default is 50 anomalies.
5) Run **test.py** by clicking run button on top right corner of screen.
6) Find segmented images and supplementary data in **UTMB_Py/predictions/data/oct** folder after test.py finishes running.
7) Anomalies may be found in **UTMB_Py/predictions/data/oct/file_path/anomalies/**. _anom.csv_ is from local averages. _cae_anomalies.csv_ is from OCT.  
8) Move data that has already been segmented from **../data/oct** to **../used_data/** or _delete from directory_ to avoid re-segmentation.

###Anomaly Detection (Alone)
If you would just like to perform anomaly detection without segmentation, follow the steps listed below.

Anomalies may be detected using OCT B-Scans as direct input **OR** 
by using local average csv files that are outputted by segmentation algorithm.
The csv file method has lower computational cost.

1. If you want to detect anomalies directly from OCT data, upload 256x512xn OCT B-Scan tif files to **UTMB_Py/data/oct folder**.
If you have csv's, upload them into **../data/csv**
2. Open **UTMB_Py/anomaly.py** in Pycharm.
3. See the following lines:
```python
if __name__ == "__main__":
	predict_anomaly(cae=True)
```
4. Default is set to using OCT scans. Set *cae=False* if csv files are used for anomaly detection.
5. Results can be found in **../predictions/{file_path}/anomalies/**
6. Use the **anom.csv** file in the ImageJ Macro **UTMB_Py/Anomaly_Viewer.ijm** when asked for a file.
When given choice between OCT vs. CSV in AnomalyViewer macro, choose OCT if the cae was used.
Otherwise, (if local averages were used for anomaly detection, vae, choose CSV).

##Contributing

Contributions to the code/algorithm are welcome. 
The deep learning model versions may be found in the **UTMB_Py/model/versions** directory.
The model version may be changed by opening **model/load.py** which contains the function that loads the saved Keras model into Python program.

Change model file path in the line of code:
```python
with open('model/versions/model_v13.json', 'r') as f:
```

Change weights file path/name in this line of code: (i.e. model_v13 to model_v14)
```python
loaded_model.load_weights("model/versions/model_v13.h5")
```

Similarly, the **anomaly detection model** may be changed by following a similar procedure in the folder **UTMB_Py/ae**. 
The file **../ae/load_cae.py** is the convolutional autoencoder (uses OCT images directly to detect anomalies) used as a default in the **test.py** and the **anomaly.py** file. 


##Author
Name: Ekin Tiu 

Email: ekintiu@gmail.com

Phone: (713)-265-8343

##Acknowledgements
Thanks to Dr. Jon Luisi, Jon Lin, and Dr. Motamedi for their guidance and contribution to this study. 



