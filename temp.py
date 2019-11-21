# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

#dummy function that does nothing
def dummy(value):
    pass

#define convolution kernels
identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])    
gaussian_kernel1 = cv2.getGaussianKernel(3,0)   #3=size, 0=standart deviation
gaussian_kernel2 = cv2.getGaussianKernel(5,0)
box_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32) / 9.0 #averaging kernel diye de geciyormus
deneme_kernel1 = np.array([[-1,-2,-1],[0,1,0],[1,2,1]])
deneme_kernel2 = np.array([[1,2,1],[2,4,1],[1,2,1]], np.float32) /9.0
deneme_kernel3 = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]],np.float32) / 273.0

#read in an image, make a grayscale copy
color_original = cv2.imread('img.jpg')
gray_original = cv2.cvtColor(color_original, cv2.COLOR_BGR2GRAY)

kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel,deneme_kernel1,deneme_kernel2,deneme_kernel3] #tum filtrelerimiz bunun icinde

#create the UI\ window and trackbars
cv2.namedWindow('Instagram Filtereleri')
#arguments: trackbarName, windowName, value (initial value), count (max value), onChange (event handler)
cv2.createTrackbar('contrast', 'Instagram Filtereleri', 50, 100, dummy)
#name=brightness, windowName, initial value=50, max value=100, event handler=dummy
cv2.createTrackbar('brightness', 'Instagram Filtereleri', 50, 100, dummy)
cv2.createTrackbar('filter', 'Instagram Filtereleri', 0, len(kernels)-1,dummy)
cv2.createTrackbar('grayscale', 'Instagram Filtereleri', 0, 1, dummy)

#main UI loop
count = 1
while True:
    #read all of the trackbar values
    grayscale = cv2.getTrackbarPos('grayscale', 'Instagram Filtereleri')
    contrast = cv2.getTrackbarPos('contrast', 'Instagram Filtereleri')
    brightness = cv2.getTrackbarPos('brightness', 'Instagram Filtereleri')
    kernel_idx = cv2.getTrackbarPos('filter','Instagram Filtereleri')
    #apply the filters
    color_modified = cv2.filter2D(color_original, -1, kernels[kernel_idx])  #filter2D kernelleri calistirma fonksiyonudur. (orjinal resim, resimdeki kanal sayisi(rgb icin 3, -1 yazarsak secilen resminkini aliyor))
    gray_modified = cv2.filter2D(gray_original, -1, kernels[kernel_idx])
    
    #apply the brightness and contrast
    color_modified = cv2.addWeighted(color_modified, contrast/50, np.zeros_like(color_original), 0, brightness - 50)
    gray_modified = cv2.addWeighted(gray_modified, contrast/50, np.zeros_like(gray_original), 0, brightness - 50)

    #wait for key press(100 milliseconds
    key = cv2.waitKey(100)
    if key == ord('q'): #int i karaktere donusturuyor
        break
    elif key == ord('s'):
        #save image
        if grayscale ==0:
            cv2.imwrite('output-{}.png'.format(count), color_modified)
        else:
            cv2.imwrite('output-{}.png'.format(count), gray_modified)
        count+=1
        
    
    #show the image
    if grayscale == 0:
        cv2.imshow('Instagram Filtereleri', color_modified)
    else:
        cv2.imshow('Instagram Filtereleri', gray_modified)


#window cleanup
cv2.destroyAllWindows()