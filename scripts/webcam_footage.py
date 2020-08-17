# For webcam
import cv2
import time

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

############################
##### WEBCAM FUNCTIONS #####
############################

def get_frame(cam):
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    return frame

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
    x2, y2 = x1 + width, y1 + height
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


    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        try:
            t1 = time.time()
            frame = get_frame(cam)
            face, bbox = extract_face(frame)
            x1, y1, x2, y2 = bbox
            img2 = cv2.rectangle(frame, (x1, y1), (x2, y2), color = (0,255,0), thickness = 3)

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


if __name__ == "__main__":
    main()