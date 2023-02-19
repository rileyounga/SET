import datetime
from PIL import Image
import cv2
from SET_Solver import main as solver
from SET_Classifier import MultiClassCNN

# global variable
title = str(datetime.datetime.now().month) + "_" + str(datetime.datetime.now().day) + "_game.png"


def SET_capture():
    cv2.namedWindow("SET")
    # default is 640x480 1.6125 wider, 1.0435 taller
    cv2.resizeWindow("SET", 640, 480)
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("SET", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
        if key == ord('c'):
            # take a screenshot and save it to img
            img = cv2.resize(frame, (258 * 4, 167 * 3))
            cv2.imwrite(title, img)
            break

    vc.release()
    cv2.destroyWindow("SET")


def main():
    # SET_capture()
    solver(title)


if __name__ == "__main__":
    main()
