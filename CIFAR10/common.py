import tkinter as tk
from mss import mss
import cv2
import numpy as np

def videoCapture(winName, winwidth, winHeight, callback):
    videoCap = cv2.VideoCapture(0)
    cv2.namedWindow(winName)
    
    while(True):
        ret, frame = videoCap.read()
        if ret==True:
            frame = cv2.flip(frame, 1)
            (h, w, _) = frame.shape
            cx, cy = int((w-winwidth)/2), int((h-winHeight)/2)
            frame = frame[cy:cy+winHeight, cx:cx+winHeight]
            callback(frame)

        key = cv2.waitKey(1)
        if key != -1 and key != 255:
            videoCap.release()
            cv2.destroyAllWindows()
            break

def screenCapture(winName, winwidth, winHeight, callback):
    window = tk.Tk()
    geo = str(winwidth)+'x'+str(winHeight)
    window.geometry(geo)
    window.wm_attributes('-alpha', 0.1)

    cv2.namedWindow(winName)
    with mss() as sct:
        while True:
            window.update()
            [_, left, top] = window.geometry().split("+")
            img = sct.grab({'left':int(left)+8, 'top':int(top)+30, 'width': winwidth, 'height': winHeight})
            frame = np.array(img)[:,:,0:3]
            callback(frame)

            key = cv2.waitKey(1)
            if key != -1 and key != 255:            
                cv2.destroyAllWindows()
                window.quit()
                break