# FaceRecognition

## General Description

This repo contains our work for the IPD446 course project: FaceNet-based Face Recognition System.

The project is organized in two parts:

1) Benchmarking (finished): in this step we took FaceNet pre-trained model and do a sensitivity analysis to see if the system
behaves as good as stated in the state-of-the-art original paper.
Paper link: https://arxiv.org/pdf/1503.03832.pdf

The system works as follows:

	a) Face Detection: Using the MTCNN library.
	b) Face Features Extraction: Using a FaceNet pre-trained model.
	c) Face Recognition: using different threshold values to compare embeddings (from 0.1 to 1.0).

2) Final Algorithm (finished): incorporate this feature extravtor into a more productive or real workflow. With the use of the Face Detection, Face Features Extraction and Classifier modules, and using real pictures, the idea is to provide an offline and online inference modules to finally prove that the detector actually works in known and unknown scenarios with both known and unknown people.

The online inference is made using a camera (e.g. laptop's webcam) in which we used the following methods to prove and compare the results and speed of results:

	a) MeanShift
	b) CAMShift
	c) Online MTCNN

## Our Code (Stage 1 - Benchmarking - Finished)

Our implementation can be found in `notebooks/`. The content is:

* Experiments_no_training.ipynb: Data generation and plot for all the experiments related to non-trained models (Default v/s MaxPool no-retrain).
* Experiments_Retraining: Data generation and plot for all the experiments related to re-trained models (MaxPool+Adam v/s MaxPool+Dropout+Adam
* Sensitivity_analysis: General procedures used to get the data from the datasets (face extraction and feature extraction) and procedures to rebuild the models we used here (no-retrainable and re-trainable models).

## Our Code (Stage 2 - Final Algorithm - Finished) 

Our implementation can be found in `notebooks/` and in `scripts/`. The content is:

* notebooks/Classifier-5-celebrities.ipynb: implementation of a SVM classifier using as training and validation sets the dataset "5-celebrities-faces-dataset". Obtention of metrics is also included.
* notebooks/Classifier-LFW.ipyng: implementation of a SVM classifier using as training and validation sets the dataset "custom-LFW" (custom LFW made by us, also available in `data/datasets/`. Obtention of metrics is also included.
* notebooks/Customized_classifier.ipynb: implementation of a NN classifier using as training and validation sets the dataset "5-celebrities-faces-dataset", "custom-LFW" and "personal-dataset". The analysis is separated by dataset
* notebooks/Feature_extraction_classifier (SVM / personal): feature extraction of the faces in every dataset we used.
* notebooks/Inference_5-celeb.ipynb: using the trained classifiers in the "Classifier" notebooks, here we use the validation set to see if the inferences are made correctly using the dataset "5-celebrities-faces-dataset".
* notebooks/Inference_LFW.ipynb: using the trained classifiers in the "Classifier" notebooks, here we use the validation set to see if the inferences are made correctly using the dataset "custom-LFW".
* notebooks/Inference_NN_classifier.ipynb: using the trained NN classifier, here we use the validation set to see if the inferences are made correctly using the dataset "custom-LFW".
* notebooks/Inference_personal.ipynb: using the trained classifiers in the "Classifier" notebooks, here we use the validation set to see if the inferences are made correctly using the dataset "custom-LFW".
* notebooks/Multiple_detection.ipynb: using the trained classifiers in the "Classifier" notebooks, here we use the validation set to see if the inferences are made correctly using the dataset "custom-LFW" in the scenario of multiple detection of people in the images.
* scripts/webcam_footage.py: Implementation of Online inference. Here we take frames of the laptop's webcam and pass it through our architecture. The user can choose which method to use for face tracking and cropping (MeanShift, CAMShift or Online MTCNN), along with the model to check the inferences (SVM or NN). For a better explanation of the code, the code itself is commented in every line so you can understand what is going on at every stage of the execution.


## Dependencies

- Language used: Python 3.7.x

* General Dependencies:
```console
jose@Amethyst:~$ sudo apt install build-essential cmake unzip pkg-config
jose@Amethyst:~$ sudo apt install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
jose@Amethyst:~$ sudo apt install libjpeg-dev libpng-dev libtiff-dev
jose@Amethyst:~$ sudo apt install libxvidcore-dev libx264-dev
jose@Amethyst:~$ sudo apt install libgtk-3-dev
jose@Amethyst:~$ sudo apt install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
jose@Amethyst:~$ sudo apt install libhdf5-serial-dev
jose@Amethyst:~$ sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
jose@Amethyst:~$ sudo apt install python3-dev python3-tk python-imaging-tk
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

## Implementation made by: 
- Patricia Franco Troya (UTFSM)
- José Luis Gallardo P. (UTFSM)
