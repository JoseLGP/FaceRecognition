zimport cv2
import time

def get_frame(cam):
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    return frame

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
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Camera", frame)
            t2 = time.time()
            framespeed = 1/(t2-t1)
            print("FPS: " + str(framespeed))
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