#!/usr/bin/env python 
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from Adafruit_BNO055 import BNO055

def absAngles(relAz,relEl):
    """
    Takes in camera relative rotations, and returns absolute rotations
    using the IMU
    """
    
    heading, roll, pitch = BNO055.read_euler()
    
    # Use the roll to transform the relative azimuth and elevation in
    # the image into absolute azimuth and elevation (no roll)
    camAz = relAz*cos(roll) - relEl*sin(roll)
    camEle = relAz*sin(roll) - relEl*cos(roll)
    
    return (heading + camAz, pitch + camEle)

def getPixelAngles(x,y,w,h,hfov,vfov):
    """
    Takes a pixel location, image size, and fields of view to locate 
    relative azimuth and elevation from camera plane to pixel
    """
    
    az = (w/2-x)/w * hfov
    el = (h/2-y)/h * vfov
    
    return (az,el)

def findLightSources(frame,threshold):
    """
    Takes a frame and a threshold, and returns the 
    minimum enclosing circle of all light sources (above threshold brightness)
    """
    
    # load the image, convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)   # Blurring eliminates noise

    # threshold the image to reveal light regions in the blurred image
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]

    # perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    sources = []
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
##        if label == 0:
##            continue

        # otherwise, construct the label mask and count the number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
        if numPixels < 250000:     # 300 pixels for large blob (arbitary, needs experimentation)
            mask = cv2.add(mask, labelMask)
    
            # find the contours in the mask, then sort them from left to right
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            cnts = contours.sort_contours(cnts)[0]
         
            # loop over the contours, finding the center of each source
            for (i, c) in enumerate(cnts):
                # Bound the sources
                (x, y, w, h) = cv2.boundingRect(c)
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                sources.append([(cX, cY), radius,(x,y,w,h)])
 
    # return the list of sources
    return sources

if __name__ == "__main__":
    
    # Pic Properties
    picWidth = 640
    picHeight = 480
    framerate = 32

    # Camera FOV
    hfov = 62.2
    vfov = 48.8
    
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (picWidth, picHeight)
    camera.framerate = framerate
    camera.vflip = True
    rawCapture = PiRGBArray(camera, size=(picWidth, picHeight))

    # allow the camera to warmup
    time.sleep(0.1)

    # capture frames from the camera
    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        frame = f.array
        
        # Find the lightSources in the unwarped image, and label them
        lightSources = findLightSources(frame, 200)     # Threshold of 200 (arbitrary, needs experimentation)
        lightLoc = []
        angles = []
        for each in lightSources:
            cX = each[0][0]
            cY = each[0][1]
            x = each[2][0]
            y = each[2][1]
            lightLoc.append((cX,cY))
            # Get azimuth and elevation
            angles.append(getPixelAngles(cX,cY,picWidth,picHeight,hfov,vfov))
            radius = each[1]
            # Encircle each light source
            cv2.circle(frame, (int(cX), int(cY)), int(radius),
                (0, 0, 255), 3)
            # Add the azimuth and elevation angles near the circle
            cv2.putText(frame, '('+str(angles[-1][0])+','+str(angles[-1][1])+')', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
