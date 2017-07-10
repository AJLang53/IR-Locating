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
import math
import threading

class getOrientation:

    def __init__(self, sensor):
        self.sensor = sensor
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.prevPitch = 0.0
        self.prevRoll = 0.0
        self.prevYaw = 0.0
        self.end = False
        if self.sensor == 'hat':
            thread = threading.Thread(target=self.runHat,args=())
        elif self.sensor == 'BNO':
            thread = threading.Thread(target=self.runBNO,args=())
            
        thread.deamon = True
        thread.start()


    def runBNO(self):
        bno = BNO055.BNO055(rst=18) 
        if not bno.begin():
            raise RuntimeError("Failed to initialize BNO055")
        
        # Print system status and self test result.
        status, self_test, error = bno.get_system_status()
        print('System status: {0}'.format(status))
        print('Self test result (0x0F is normal): 0x{0:02X}'.format(self_test))
        # Print out an error if system status is in error mode.
        if status == 0x01:
            print('System error: {0}'.format(error))
            print('See datasheet section 4.3.59 for the meaning.')

        # Print BNO055 software revision and other diagnostic data.
        sw, bl, accel, mag, gyro = bno.get_revision()
        print('Software version:   {0}'.format(sw))
        print('Bootloader version: {0}'.format(bl))
        print('Accelerometer ID:   0x{0:02X}'.format(accel))
        print('Magnetometer ID:    0x{0:02X}'.format(mag))
        print('Gyroscope ID:       0x{0:02X}\n'.format(gyro))

        while not self.end:
            
            yaw, roll, pitch = bno.read_euler()
            
            while pitch > 180:
                pitch -= 360
            while roll > 180:
                roll -= 360
            while yaw > 180:
                yaw -= 360

            if not (pitch <= -360):
                self.pitch = pitch

            if not (roll <= -360):
                self.roll = roll

            if not (yaw <= -360):
                print(yaw)
                self.yaw = yaw

##            self.pitch = self.lowPass(pitch, self.prevPitch, 50 , 10)
##            self.roll = self.lowPass(roll, self.prevRoll, 50 , 10)
##            self.yaw = self.lowPass(yaw, self.prevYaw, 50 ,10)
##
            time.sleep(0.1)
##
##            self.prevPitch = self.pitch
##            self.prevRoll = self.roll
##            self.prevYaw = self.yaw
                
        
    def runHat(self):
        
        senseHat = SenseHat()
        samples = 5

        while not self.end:
            cumPitch = 0
            cumRoll = 0
            cumYaw = 0
            for i in range(samples):
                orient = senseHat.get_orientation()
                cumPitch += orient['pitch']
                cumRoll += orient['roll']
                cumYaw += orient['yaw']

            pitch = cumPitch/samples
            roll = cumRoll/samples
            yaw = cumYaw/samples
            
            while pitch > 180:
                pitch -= 360
            while roll > 180:
                roll -= 360
            while yaw > 180:
                yaw -= 360

            self.pitch = self.lowPass(pitch, self.prevPitch, 50 , 10)
            self.roll = self.lowPass(roll, self.prevRoll, 50 , 10)
            self.yaw = self.lowPass(yaw, self.prevYaw, 50 ,10)

            time.sleep(0.01)

            self.prevPitch = self.pitch
            self.prevRoll = self.roll
            self.prevYaw = self.yaw

    def lowPass(self,x, y_old, rc, dt):
        alpha = float(dt)/(rc+ dt)
        return alpha*x + (1- alpha)*(y_old)

def getAbsAngles(relAz,relEl,IMU):
    """
    Takes in camera relative rotations, and returns absolute rotations
    using the IMU
    """

    pitch = orient.pitch
    roll = orient.roll
    yaw = orient.yaw
        
    # Use the roll to transform the relative azimuth and elevation in
    # the image into absolute azimuth and elevation (no roll) (I'm not sure if
    # this is correct)
    camAz = relAz*math.cos(roll*180/math.pi) - relEl*math.sin(roll*180/math.pi)
    camEle = relAz*math.sin(roll*180/math.pi) - relEl*math.cos(roll*180/math.pi)
    
    return (yaw + camAz, pitch + camEle)
        

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

    ap = argparse.ArgumentParser()
    ap.add_argument("--imu",required=False, help="hat or bno")
    args = vars(ap.parse_args())

    useHat = False
    useBNO = False

    if args["imu"]:
        if args["imu"].lower() == 'hat':
            useHat = True
            from sense_hat import SenseHat
            orient = getOrientation('hat')
        if args["imu"].lower() == 'bno':
            useBNO = True
            from Adafruit_BNO055 import BNO055
            orient = getOrientation('BNO')
            
        
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
        lightSources = findLightSources(frame, 230)     # Threshold of 200 (arbitrary, needs experimentation)
        lightLoc = []
        angles = []
        trueAngles = []
        for each in lightSources:
            cX = each[0][0]
            cY = each[0][1]
            x = each[2][0]
            y = each[2][1]
            lightLoc.append((cX,cY))
            
            # Get azimuth and elevation
            relAngs = getPixelAngles(cX,cY,picWidth,picHeight,hfov,vfov)
            angles.append(relAngs)
            if useHat:
                trueAngles.append(getAbsAngles(relAngs[0], relAngs[1], "Hat"))
            elif useBNO:
                trueAngles.append(getAbsAngles(relAngs[0], relAngs[1], "BNO"))
            radius = each[1]
            # Encircle each light source
            cv2.circle(frame, (int(cX), int(cY)), int(radius),
                (0, 0, 255), 3)
            # Add the azimuth and elevation angles near the circle
            if useHat or useBNO:
                cv2.putText(frame, '('+str(round(trueAngles[-1][0],2))+','+str(round(trueAngles[-1][1],2))+')', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                cv2.putText(frame, '('+str(round(angles[-1][0],2))+','+str(round(angles[-1][1],2))+')', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            orient.end = True
            break
