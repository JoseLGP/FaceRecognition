# For webcam
import cv2
import time
import os
import sys

# For Face Embedding Jobs
from matplotlib import pyplot
from numpy import savez_compressed, asarray, load
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from numpy import load
from numpy import expand_dims
from PIL import Image
from ipywidgets import FileUpload
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import joblib
import cv2


def get_frame(cam):
    """ Function that takes the cam input and outputs the frame and it's bool flag

    Params:
    - cam: cv2.VideoCapture()

    Returns:
    - frame: np.array() with img data
    - ret: True, if frame is read correctly. False, otherwise

    """
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    return frame, ret


# extract a single face from a given photograph
def extract_face(frame, required_size=(160, 160)):
    """ Extract a single face from a given photograph or video-frame

    Params:
    - frame: img as np.array() - Taken from video input or a local img
    - required_size: (width, height) as tuple - Desired face crop output size

    Outputs:
    - face_array: img as np.array() - Is an img with the face crop, just the face
    - bbox_dims: tuple of values (x, y, width, height) - dimensions of the bounding box - where the face is located

    """

    # Convert to array
    pixels = asarray(frame)

    # Create the detector, using default weights
    detector = MTCNN()

    # Detect faces in the image
    results = detector.detect_faces(pixels)

    # Extract the bounding box from the first face
    try: 
        x1, y1, width, height = results[0]['box']

        # Bug fix - Sometimes the detector takes as negative values the x and y coords
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + abs(width), y1 + abs(height)

        # Extract the face
        face = pixels[y1:y2, x1:x2]

        # Resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

        # Get bbox dims
        bbox_dims = [x1, y1, x2, y2]
        return face_array, bbox_dims

    # If face is not found, return the entire frame again
    except:
        face_array = np.asarray(frame)
        bbox_dims = [0., 0., frame.cols, frame.rows]


def get_embedding(model, face_pixels):
    """ Get the face embedding for one face

    Params:
    - model: The model which will get the embeddings. In this case, is FaceNet
    - face_pixels: Face crop

    Outputs:
    - yhat[0]: Prediction of the embedding - It's just one face so we take the first result

    """

    # Scale pixel values to float
    face_pixels = face_pixels.astype('float32')

    # Standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # Transform face into one sample
    samples = expand_dims(face_pixels, axis=0)

    # Make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0] 
      

def main():
    """ MeanShift demonstration to track face and predict using SVM and NNs
    """

    # Get webcam footage
    cam = cv2.VideoCapture(-1)

    # Params to load models
    mode = "personal"
    facenet_path = "../models/facenet_keras.h5"

    if mode == "personal":
        model_name = "../models/svc3.sav"
        labels = ("Jose_Luis", "Patricia", "Unknown")

    # Loading the models
    clf = joblib.load(model_name)
    emb_extractor = load_model(facenet_path)
    emb2 = list()


    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    # Get actual footage from webcam
    frame, ret = get_frame(cam)

    # Call MTCNN to get initial face
    face, bbox = extract_face(frame)

    # Bounding Box dimensions
    x1, y1, x2, y2 = bbox
    track_window = (x1, y1, x2, y2)

    # Region to track
    roi = frame[x1:x2, y1:y2]

    # MeanShift computations
    # 1) Get Histogram of the desired region 
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 2) Apply threshold to filter correct values
    mask = cv2.inRange(hsv_roi, np.array([0., 60., 32. ]), np.array([180., 255., 255.]))
    # 3) Get histogram of desired region
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    # 4) Scale pixel values to 0-255
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # 5) Convergence criterias
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # For every new videoframe captured by the webcam
    while True:
        try:
            # Take the time to measure FPS
            t1 = time.time()

            # Capture video frame
            frame, ret = get_frame(cam)

            # Get a copy of the actual frame
            frame_real = np.array(frame)

            # If videoframe was captured correctly
            if ret == True:

                # Again, do MeanShift computations but to refresh the tracking window
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Histogram backprojection for MeanShift
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # Here we get the new tracking window
                ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                x, y, w, h = track_window

                # Draw the tracking region in out main videoframe
                img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

                # Get a copy of the tracking region
                tracking = frame[y:y+h, x:x+w] 

                # Prediction tasks
                # 1) Get embedding with FaceNet
                emb = get_embedding(emb_extractor, tracking)
                emb2 = list()
                emb2.append(emb)
                emb2 = np.asarray(emb2)
                # 2) Call SVM or NN classifier for online predictions
                y_class = clf.predict(emb2)
                y_probs = clf.predict_proba(emb2)
                class_index = y_class[0]
                class_proba = y_probs[0, class_index] * 100
                print("Class_index: ", class_index, " - Subject: ", str(labels[class_index]))
                print("Probs: ", y_probs)
                # 3) Draw text in the main videoframe with the class predicted
                img3 = cv2.putText(frame, str(labels[class_index]), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 3)

            # Result preview
            cv2.imshow("Camera", frame)
            cv2.imshow("FRAME REAL", frame_real)
            cv2.imshow("Tracking", tracking)
            cv2.imshow("Face found", face)

            # Again, to measure speed of the algorithm
            t2 = time.time()
            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
            print(type(frame))

            # Press ESC key to exit
            c = cv2.waitKey(1)
            if c == 27:
                break

        # In case of any error, release the webcam and destroy all preview windows !!
        except:
            cam.release()
            cv2.destroyAllWindows()

    #cam.release()
    #cv2.destroyAllWindows()
    return

def main2():
    """ CAMShift demonstration to track face and predict using SVM and NNs

    - It's an iteration of the one using MeanShift (previous one), but here we 
      update the size of the tracking window

    """

    # Get webcam footage
    cam = cv2.VideoCapture(0)

    # Params to load models
    mode = "personal"
    facenet_path = "../models/facenet_keras.h5"

    if mode == "personal":
        model_name = "../models/svc3.sav"
        labels = ("Jose_Luis", "Patricia", "Unknown")
        #model_name = "../models/svc4.sav"
        #labels = ("George_HW_Bush","Jose_Luis", "Leonardo_DiCaprio", "Meryl_Streep", "Michael_Jackson", "Patricia", "Unknown")
        #model_name = "../models/nn_clf.h5"
        #labels = ("George_HW_Bush","Jose_Luis", "Leonardo_DiCaprio", "Meryl_Streep", "Michael_Jackson", "Patricia", "Unknown")
        

    # Loading the models
    clf = joblib.load(model_name)
    #clf = load_model("../models/nn_clf.h5")
    emb_extractor = load_model(facenet_path)
    emb2 = list()

    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    # Capture first frame
    frame, ret = get_frame(cam)

    # Initial processing for CAMShift
    # 1) If capture is successful
    if ret == True:
        cv2.imshow("INITIAL FRAME", frame)
        # 2) MTCNN call for the initial face
        face, bbox = extract_face(frame)
        # 3) Initial tracking window dimensions
        x1, y1, x2, y2 = bbox
        track_window = (x1, y1, x2, y2)
        print(track_window)
        # 4) A bit of time-space for MTCNN processing
        time.sleep(2)
        # 5) CAMShift processing
        # 5.1) Crop of desired tracking region
        roi = frame[x1:x2, y1:y2]
        # 5.2) BGR to HSV space color for histogram
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        cv2.imshow("DETECTED FACE CROP", roi)
        cv2.imshow("Face found", face)
        cv2.imshow("DETECTED FACE CROP HSV", roi_hsv)
        cv2.imshow("Face found HSV", face_hsv)
        # 5.3) Thresholding HSV values
        mask = cv2.inRange(face_hsv, np.array([0., 60., 32. ]), np.array([180., 255., 255.]))
        # 5.4) Histogram generation
        roi_hist = cv2.calcHist([face_hsv], [0], mask, [180], [0, 180])
        # 5.5) Scale pixel values from 0 to 255
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # 5.6) Convergence criterias
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


    # For every new videoframe captured by the webcam
    while True:
        try:
            # Take the time to measure FPS
            t1 = time.time()

            # Capture video frames
            frame, ret = get_frame(cam)

            # Generate a copy of the actual video frame
            frame_real = np.array(frame)

            # If videoframe was captured correctly
            if ret == True:

                # CAMShift operations
                # 1) BGR to HSV actual video frame
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # 2) Histogram backprojection of the desired region to track
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # 3) CAMShift
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                # Get a set of points for the updated tracking region
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                # Draw in our actual frame the Tracking window
                img2 = cv2.polylines(frame, [pts], True, 255, 2)

                # Get dimensions for the Tracking window
                print("Box dims: ", track_window)
                xf, yf, wf, hf = track_window

                # Get the actual Tracking region
                frame2 = frame_real[yf:yf+hf, xf:xf+wf]


                # Prediction tasks
                #frame2 = cv2.resize(frame2, (160,160))

                # 1) Get embedding with FaceNet
                emb = get_embedding(emb_extractor, frame2)
                emb2 = list()
                emb2.append(emb)
                emb2 = np.asarray(emb2)
                # 2) Call SVM or NN classifier for online predictions
                y_class = clf.predict(emb2)
                y_probs = clf.predict_proba(emb2)
                class_index = y_class[0]
                class_proba = y_probs[0, class_index] * 100
                print("Class_index: ", class_index, " - Subject: ", str(labels[class_index]))
                print("Classes: ", y_class)
                print("Probs: ", y_probs)
                #print("Proba: ", class_proba)
                #print(y_class)
                #print(y_probs)
                #print(emb2)
                # 3) Draw text in the main videoframe with the class predicted
                img3 = cv2.putText(frame, str(labels[class_index]), (xf, yf-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness = 3)                

            # Result preview
            cv2.imshow("Camera", frame)
            cv2.imshow("Frame real", frame_real)
            #cv2.imshow("To track", region_to_track)
            cv2.imshow("CamShift", hsv)
            cv2.imshow("Shifting", frame2)
            #cv2.imshow("Face detected", face)

            # Again, to measure speed of the algorithm
            t2 = time.time()
            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
            print(type(frame))

            # Press ESC key to exit
            c = cv2.waitKey(1)
            if c == 27:
                break

        # In case of any error, release the webcam and destroy all preview windows !!
        except:
            cam.release()
            cv2.destroyAllWindows()

    #cam.release()
    #cv2.destroyAllWindows()
    return


def main3():
    """ Full Precision demonstration to track face and predict using SVM or NNs

    - Here we do calls to the MTCNN Face Detector Module for every new frame we capture
      for the videocam
    - Here we don't use MeanShift or CAMShift, just several calls to MTCNN

    """

    # Get webcam footage
    cam = cv2.VideoCapture(0)

    # Params to load models
    mode = "personal"
    facenet_path = "../models/facenet_keras.h5"

    if mode == "personal":
        model_name = "../models/svc3.sav"
        labels = ("Jose_Luis", "Patricia", "Unknown")

    # Loading the models
    clf = joblib.load(model_name)
    emb_extractor = load_model(facenet_path)
    emb2 = list()

    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    # Get actual footage from webcam
    frame, ret = get_frame(cam)

    # Call MTCNN to get initial face
    face, bbox = extract_face(frame)

    # Time-window to let MTCNN process
    time.sleep(2)

    # Bounding box dimensions
    x1, y1, x2, y2 = bbox

    # For every new videoframe captured by the webcam
    while True:
        try:
            # Take the time to measure FPS
            t1 = time.time()

            # Capture video frame
            frame, ret = get_frame(cam)

            # If videocapture was successful
            if ret == True:
                print("Extracting face ...")

                # Call MTCNN to get initial face
                face, bbox = extract_face(frame)

                #time.sleep(2)
                # Get bounding box dimensions
                x1, y1, x2, y2 = bbox

                # Draw bounding box in our actual frame
                img2 = cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

                # Prediction tasks
                # 1) Get embedding with FaceNet
                emb = get_embedding(emb_extractor, face)
                emb2 = list()
                emb2.append(emb)
                emb2 = np.asarray(emb2)
                # 2) Call SVM or NN classifier for online predictions
                y_class = clf.predict(emb2)
                y_probs = clf.predict_proba(emb2)
                class_index = y_class[0]
                #class_proba = y_probs[0, class_index] * 100
                print("Class_index: ", class_index, " - Subject: ", str(labels[class_index]))
                print("Classes: ", y_class)
                print("Probs: ", y_probs)
                # 3) Draw text in the main videoframe with the class predicted
                img3 = cv2.putText(frame, str(labels[class_index]), (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness = 3)


            # Result preview
            cv2.imshow("Camera", frame)
            cv2.imshow("Face found", face)

            # Again, to measure speed of the algorithm
            t2 = time.time()
            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
            print(type(frame))

            # Press ESC key to exit
            c = cv2.waitKey(1)
            if c == 27:
                break

        # In case of any error, release the webcam and destroy all preview windows !!
        except:
            cam.release()
            cv2.destroyAllWindows()

    #cam.release()
    #cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    mode = 3 # 1: MeanShift - 2: CamShift - 3: online MTCNN

    if mode == 1:
        main() # MeanShift
    if mode == 2:        
        main2() # CAMShift
    if mode == 3:
        main3() # online MTCNN
