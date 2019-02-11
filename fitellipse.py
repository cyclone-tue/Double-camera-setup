import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

cap = cv2.VideoCapture(0)

def threshold(image,values):
    low_h, low_s, low_v, high_h, high_s, high_v = values
    imconv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if (high_h < low_h):
        output1 = cv2.inRange(imconv, (low_h,low_s,low_v),(179, high_s, high_v))
        output2 = cv2.inRange(imconv, (0,low_s,low_v),(high_h, high_s, high_v))
        output = output1+output2        
    else:
        output=cv2.inRange(imconv, (low_h,low_s,low_v),(high_h, high_s, high_v))
    return output

def weirdFilter(image):
    kernel = np.ones((7,7),np.float32)
    return cv2.filter2D(image,-1,kernel)

def lowPass(image,cutoff):
    shape = np.shape(image)
    pad_shape = np.multiply(shape,(1-cutoff)/2).astype(int)
    center = np.ones(np.multiply(shape,cutoff).astype(int),np.float32)
    center = np.pad(center,((pad_shape[0],pad_shape[0]),(pad_shape[1],pad_shape[1])),'constant',constant_values=0)
    magnitude_spectrum = cv2.normalize(center,0,255,cv2.NORM_MINMAX)
    cv2.imshow("filter",magnitude_spectrum)
    return np.multiply(image,np.pad(center,((0,np.shape(image)[0]-np.shape(center)[0]),(0,np.shape(image)[1]-np.shape(center)[1])),'constant',constant_values=0))
    
while True:
    _,frame = cap.read()

    imred =     threshold(frame,[160, 32, 160, 10,  150, 255]) #red filter
    imgreen =   threshold(frame,[ 70, 32, 160, 90, 150, 255]) #green filter
    imblue =    threshold(frame,[110, 32, 160, 140, 150, 255]) #blue filter

    A1 = cv2.bitwise_and(weirdFilter(imred),weirdFilter(imgreen))
    A2 = cv2.bitwise_and(weirdFilter(imred),weirdFilter(imblue))
    A3 = cv2.bitwise_and(weirdFilter(imgreen),weirdFilter(imblue))
    
    B = cv2.bitwise_and(A1,imblue)
    G = cv2.bitwise_and(A2,imgreen)
    R = cv2.bitwise_and(A3,imred)

    RGB = cv2.bitwise_or(R,G)
    RGB = cv2.bitwise_or(RGB,B)

    combined = np.stack([imblue,imgreen,imred],axis=2)
    
    _, contours, hierarchy = cv2.findContours(RGB,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if (len(contours) > 20):
        points = np.vstack(contours)
        fit=cv2.fitEllipse(points)
        ellipse=cv2.ellipse(frame,fit,(255,0,0),5)

    cv2.imshow("rgb",combined)
    cv2.imshow("filtered",weirdFilter(combined))
    cv2.imshow("overlap",RGB)
    cv2.imshow("Frame+ellipse", frame)

    """
    #cv2.imshow("Red" , imred)
    f = np.fft.fft2(imred)
    fshift_red = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift_red))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum,0,255,cv2.NORM_MINMAX)
    #cv2.imshow("Red Fourier", magnitude_spectrum)

    #cv2.imshow("Blue" , imblue)
    f = np.fft.fft2(imblue)
    fshift_blue = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift_blue))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum,0,255,cv2.NORM_MINMAX)
    #cv2.imshow("Blue Fourier", magnitude_spectrum)

    cv2.imshow("Green" , imgreen)
    f = np.fft.fft2(imgreen)
    fshift_green = np.fft.fftshift(f)
    #fshift_green = lowPass(fshift_green,1/2)
    inverse = np.fft.ifftshift(fshift_green)
    inverse = np.fft.ifft(inverse)
    cv2.imshow("Green Filtered", cv2.normalize(np.abs(inverse), 0, 255, cv2.NORM_MINMAX))
    magnitude_spectrum = 20 * np.log(np.abs(fshift_green)+1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum,0,255,cv2.NORM_MINMAX)
    cv2.imshow("Green Fourier", magnitude_spectrum)

    #print(magnitude_spectrum)
    #multiplied = fshift_green
    multiplied = np.multiply(fshift_blue,fshift_green)
    multiplied = np.multiply(multiplied,fshift_red)
    multiplied = lowPass(multiplied,1/3)

    #magnitude_spectrum = 20 * np.log(np.abs(multiplied)+1)
    magnitude_spectrum = cv2.normalize(np.abs(multiplied),0,255,cv2.NORM_MINMAX)
    #cv2.imshow("multiplied Fourier", magnitude_spectrum)

    inverse = np.fft.ifftshift(multiplied)
    inverse = np.fft.ifft2(inverse)
    inverse = np.abs(inverse)
    inverse = cv2.normalize(inverse,0,255,cv2.NORM_MINMAX)
    #cv2.imshow("inverse", inverse)
    """
 
    key = cv2.waitKey(1)
    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()

