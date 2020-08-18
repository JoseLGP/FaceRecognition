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

#os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

############################
##### WEBCAM FUNCTIONS #####
############################

def get_frame(cam):
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    return frame, ret

####################################
##### FACE EMBEDDING FUNCTIONS #####
####################################

# extract a single face from a given photograph
def extract_face(frame, required_size=(160, 160)):
    # load image from file
    #image = Image.open(filename)
    # convert to RGB, if needed
    #image = image.convert('RGB')
    # convert to array
    pixels = asarray(frame)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + abs(width), y1 + abs(height)
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    bbox_dims = [x1, y1, x2, y2]
    return face_array, bbox_dims

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0] 
      

def main():
    cam = cv2.VideoCapture(0)

    mode = "personal"
    facenet_path = "../models/facenet_keras.h5"

    if mode == "personal":
        model_name = "../models/svc3.sav"
        labels = ("Jose_Luis", "Patricia")

    clf = joblib.load(model_name)
    emb_extractor = load_model(facenet_path)
    emb2 = list()
    face_detected = False


    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    frame, ret = get_frame(cam)
    face, bbox = extract_face(frame)
    x1, y1, x2, y2 = bbox
    track_window = (x1, y1, x2, y2)
    roi = frame[x1:x2, y1:y2]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array([0., 60., 32. ]), np.array([180., 255., 255.]))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        try:
            t1 = time.time()
            frame, ret = get_frame(cam)

            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                ret, track_window = cv2.meanShift(dst, track_window, term_crit)

                x, y, w, h = track_window
                img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)


            #if not face_detected:
                #face, bbox = extract_face(frame)
                #face_detected = True
                #x1, y1, x2, y2 = bbox
                #region_to_track = frame[x1:x2, y1, y2]
                #region_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                #mask = cv2.inRange(region_hsv, np.array([0.,60.,32.]), np.array([180.,255.,255.]))
                #roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0.180])
                #cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                #term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                #img2 = cv2.rectangle(frame, (x1, y1), (x2, y2), color = (0,255,0), thickness = 3)

            # Prediction tasks
                #emb = get_embedding(emb_extractor, face)
                #emb2 = list()
                #emb2.append(emb)
                #emb2 = np.asarray(emb2)
                #y_class = clf.predict(emb2)
                #y_probs = clf.predict_proba(emb2)
                #class_index = y_class[0]
                #class_proba = y_probs[0, class_index] * 100
                #img3 = cv2.putText(frame, str(labels[class_index]), (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness = 3)
            #else:
                #print("Face obtained")

            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            #ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            #x,y,w,h = track_window
            #img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)


            cv2.imshow("Camera", frame)
            #cv2.imshow("To track", region_to_track)
            #cv2.imshow("CamShift", img2)
            cv2.imshow("Face found", face)
            #cv2.imshow("Face detected", face)
            t2 = time.time()



            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
            print(type(frame))
            #print("Proba: " + str(clas_proba))
            #print(emb)

            c = cv2.waitKey(1)
            if c == 27:
                break
        except:
            cam.release()
            cv2.destroyAllWindows()

    cam.release()
    cv2.destroyAllWindows()

def main2():
    cam = cv2.VideoCapture(0)

    mode = "personal"
    facenet_path = "../models/facenet_keras.h5"

    if mode == "personal":
        model_name = "../models/svc3.sav"
        labels = ("Jose_Luis", "Patricia", "Unknown")

    clf = joblib.load(model_name)
    emb_extractor = load_model(facenet_path)
    emb2 = list()
    #face_detected = False


    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    frame, ret = get_frame(cam)

    if ret == True:
        cv2.imshow("INITIAL FRAME", frame)
        face, bbox = extract_face(frame)
        x1, y1, x2, y2 = bbox
        track_window = (x1, y1, x2, y2)
        print(track_window)
        time.sleep(2)
        roi = frame[x1:x2, y1:y2]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        cv2.imshow("DETECTED FACE CROP", roi)
        cv2.imshow("Face found", face)
        cv2.imshow("DETECTED FACE CROP HSV", roi_hsv)
        cv2.imshow("Face found HSV", face_hsv)
        mask = cv2.inRange(face_hsv, np.array([0., 60., 32. ]), np.array([180., 255., 255.]))
        #mask = cv2.inRange(hsv_roi, np.array([0., 60., 32. ]), np.array([200., 255., 255.])) # Threshold
        roi_hist = cv2.calcHist([face_hsv], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        try:
            t1 = time.time()
            frame, ret = get_frame(cam)
            frame_real = np.array(frame)

            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv2.polylines(frame, [pts], True, 255, 2)
                print("Box dims: ", track_window)
                xf, yf, wf, hf = track_window

                frame2 = frame_real[yf:yf+hf, xf:xf+wf]


            # Prediction tasks
                emb = get_embedding(emb_extractor, frame2)
                emb2 = list()
                emb2.append(emb)
                emb2 = np.asarray(emb2)
                y_class = clf.predict(emb2)
                y_probs = clf.predict_proba(emb2)
                class_index = y_class[0]
                class_proba = y_probs[0, class_index] * 100
                print("Proba: ", class_proba)
                print(y_class)
                print(y_probs)
                print(emb2)
                img3 = cv2.putText(frame, str(labels[class_index]), (xf, yf-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness = 3)                
            #else:
                #print("Face obtained")

            cv2.imshow("Camera", frame)
            cv2.imshow("Frame real", frame_real)
            #cv2.imshow("To track", region_to_track)
            cv2.imshow("CamShift", hsv)
            cv2.imshow("Shifting", frame2)
            #cv2.imshow("HSV", hsv)
            #cv2.imshow("Face detected", face)
            t2 = time.time()

            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
            print(type(frame))

            c = cv2.waitKey(1)
            if c == 27:
                break

        except:
            cam.release()
            cv2.destroyAllWindows()

    cam.release()
    cv2.destroyAllWindows()


def main3():
    cam = cv2.VideoCapture(0)

    mode = "personal"
    facenet_path = "../models/facenet_keras.h5"

    if mode == "personal":
        model_name = "../models/svc3.sav"
        labels = ("Jose_Luis", "Patricia")

    clf = joblib.load(model_name)
    emb_extractor = load_model(facenet_path)
    emb2 = list()
    face_detected = False


    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    frame, ret = get_frame(cam)
    face, bbox = extract_face(frame)
    #time.sleep(2)
    x1, y1, x2, y2 = bbox

    while True:
        try:
            t1 = time.time()
            frame, ret = get_frame(cam)
            face, bbox = extract_face(frame)
            x1, y1, x2, y2 = bbox

            img2 = cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

            # Prediction tasks
            emb = get_embedding(emb_extractor, face)
            emb2 = list()
            emb2.append(emb)
            emb2 = np.asarray(emb2)
            y_class = clf.predict(emb2)
            y_probs = clf.predict_proba(emb2)
            class_index = y_class[0]
            class_proba = y_probs[0, class_index] * 100
            img3 = cv2.putText(frame, str(labels[class_index]), (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness = 3)


            cv2.imshow("Camera", frame)
            cv2.imshow("Face found", face)
            t2 = time.time()

            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
            print(type(frame))
            print("Proba: " + str(clas_proba))
            #print(emb)

            c = cv2.waitKey(1)
            if c == 27:
                break
        except:
            cam.release()
            cv2.destroyAllWindows()

    cam.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    mode = 2 # 1: MeanShift - 2: CamShift - 3: online MTCNN

    if mode == 1:
        main() # MeanShift
    if mode == 2:        
        main2() #Camshift
    if mode == 3:
        main3() # online MTCNN