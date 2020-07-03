# FaceRecognition

## General Description

This repo contains our work for the IPD446 course project: FaceNet-based Face Recognition System.

The project is organized in two parts:

1) Benchmarking (current status): in this step we took FaceNet pre-trained model and do a sensitivity analysis to see if the system
behaves as good as stated in the state-of-the-art original paper.
Paper link: https://arxiv.org/pdf/1503.03832.pdf

The system works as follows:

a) Face Detection: Using the MTCNN library 
b) Face Features Extraction: Using a FaceNet pre-trained model
c) Face Recognition: using different threshold values to compare embeddings (from 0.1 to 1.0)

2) Final Algorithm (to work on it): the idea is now to improve the (VAL,FAR) rates by the use of a multi-class classifier (e.g. SVM), and to incorporate this recognition method for online recognition (e.g. web-cam).

## Dependencies

- Language used: Python 3.7.x

* General Dependencies:
```console
jose@Amethyst:~$ sudo apt-get install build-essential cmake unzip pkg-config
jose@Amethyst:~$ sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
jose@Amethyst:~$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
jose@Amethyst:~$ sudo apt-get install libxvidcore-dev libx264-dev
jose@Amethyst:~$ sudo apt-get install libgtk-3-dev
jose@Amethyst:~$ sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
jose@Amethyst:~$ sudo apt-get install libhdf5-serial-dev
jose@Amethyst:~$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
jose@Amethyst:~$ sudo apt-get install python3-dev python3-tk python-imaging-tk
jose@Amethyst:~$ pip install opencv-contrib-python

```
Based on: https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/

* Specific Dependencies:

	* MTCNN (Multi-Task Convolutional Neural Network
		1. Link: https://github.com/ipazc/mtcnn
		2. Install: `pip install mtcnn`
	* PIL (Python Imaging Library)
		1. Link: https://pypi.org/project/Pillow/2.2.2/
		2. Install: `pip install Pillow`
	* Matplotlib:
		1. Link: https://pypi.org/project/matplotlib/
		1. Install: `pip install matplotlib`
	* Numpy:
		1. Link: https://pypi.org/project/numpy/
		2. Install: `pip install numpy`

	* Virtualenv (optional, but recommended):
		1. Link: https://pypi.org/project/virtualenv/
		2. Install: `pip install virtualenv`
	* Scipy:
		1. Link: scipy.org/install.html
		2. Install: `pip install scipy`
	* imutils, h5py, requests and progressbar2:
		1. Link: ...
		2. Install: `pip install imutils h5py requests progressbar2`
	* scikit-learn & scikit-image:
		1. Link: https://scikit-learn.org/stable/install.html
		2. Install: `pip install scikit-learn scikit-image`
	* Tensorflow:
		1. Link: https://www.tensorflow.org/install/pip
		2. Install: `pip install tensorflow`
	* Keras:
		1. Link: https://keras.io/
		2. Install: `pip install keras`

## Our Code

Our implementation can be found in `notebooks/`. The content is:

* Experiments_no_training.ipynb: Data generation and plot for all the experiments related to non-trained models (Default v/s MaxPool no-retrain).
* Experiments_Retraining: Data generation and plot for all the experiments related to re-trained models (MaxPool+Adam v/s MaxPool+Dropout+Adam
* Sensitivity_analysis: General procedures used to get the data from the datasets (face extraction and feature extraction) and procedures to rebuild the models we used here (no-retrainable and re-trainable models).


## Implementation made by: 
- Patricia Franco Troya (UTFSM)
- José Luis Gallardo P. (UTFSM)
