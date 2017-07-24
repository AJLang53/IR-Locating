# IR-Locating
Determine azimuth and elevation to light source

Uses a PiCamera module to take video, and then thresholding to find regions of high intensity, and identifies them as light sources. By using the FOV of the camera and it's resolution, the image azimuth and elevation of each light source can be determined. 

If an IMU is connected, the orientation data from the IMU can be combined with the pixel location to determine true azimuth and elevation.
