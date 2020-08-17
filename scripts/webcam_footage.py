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
    return face_array

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
      
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # set labes
        if result != result_list[0]:                                       
            ax.text(x,y - 1, "Unknown", fontsize = 13, color = 'red')
        else:
            for i in range(len(labels)):
                if class_index == i:
                    ax.text(x,y - 1,labels[i], fontsize = 13,color = 'red') 
#       cv2.putText(data, "rect", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
        
#                   name = labels[i]
#                   cv2.putText(data, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 1) 
#       if class_index != i:
#           cv2.putText(data, 'Unknown', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 1)       
#   # show the plot
#   cv2.imshow("Detected faces", data)
    #pyplot.imshow()    
    # T0 load and hold the image
#   cv2.waitKey(0)
    # To close the window after the required kill value was provided
#   cv2.destroyAllWindows()
    pyplot.show()

def main():
    cam = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        try:
            #ret, frame = cam.read()
            #frame = cv2.flip(frame, 1)
            t1 = time.time()
            frame = get_frame(cam)
            face = extract_face(frame)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Camera", frame)
            cv2.imshow("Face detected", face)
            t2 = time.time()
            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
            print(type(frame))
            print("Pause for 1 sec...")
            time.sleep(1)

            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            #cv2.imshow('Input', frame)

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