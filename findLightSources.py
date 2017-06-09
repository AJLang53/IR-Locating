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

def getPixelAngles(x,y,w,h,hfov,vfov):
    """
    Takes a pixel location, image size, and fields of view to locate 
    relative azimuth and elevation from camera plane to pixel
    """
    
    az = x/w * hfov
    el = y/h * vfov
    
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
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    # perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
 
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    
    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]
 
    # loop over the contours, finding the center of each source
    sources = []
    for (i, c) in enumerate(cnts):
        # Find the centers
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        sources.append(((cX, cY), radius))
 
    # return the list of sources
    return sources

if __name__ == "__main__":
    
    # Pic Properties
    picWidth = 640
    picHeight = 480
    framerate = 32
    
    # Camera Focal Lengths
    f_x = 
    f_y = 
    
    # Camera Center Coordinates
    c_x = 
    c_y = 
    
    # Construct camera matrix
    camMatrix = np.array([[f_x, 0., c_x],
                          [0., f_y, c_y],
                          [0., 0., 1.]])
    
    # Distortion Matrix
    distortMatrix = 5.44787247e-02, 1.23043244e-01, -4.52559581e-04, 5.47011732e-03, -6.83110234e-01
    
    # Generate optimal camera matrix
    newCamMatrix, roi = cv2.getOptimalNewCameraMatrix(camMatrix, distortMatrix, (picWidth, picHeight), 0)

    # Generate LUTs for undistortion
    CamMapX, CamMapY = cv2.initUndistortRectifyMap(camMatrix, distortMatrix, None, newCamMatrix,
                                                         (picWidth, picHeight), 5)
    
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (picWidth, picHeight)
    camera.framerate = framerate
    rawCapture = PiRGBArray(camera, size=(picWidth, picHeight))

    # allow the camera to warmup
    time.sleep(0.1)

    # capture frames from the camera
    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        frame = f.array
        
        # Unwarp the image
        unwarpedFrame = cv2.remap(frame, camMapX, camMapY, cv2.INTER_LINEAR).copy()

        # Find the lightSources in the unwarped image, and draw a circle around them
        lightSources = findLightSources(unwarpedFrame)
        
        for each in lightSources:
            cX = each[0][0]
            cY = each[0][1]
            radius = each[1]
            cv2.circle(unwarpedFrame, (int(cX), int(cY)), int(radius),
                (0, 0, 255), 3)

        # show the frame
        cv2.imshow("Unwarped Frame", unwarpedFrame)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
