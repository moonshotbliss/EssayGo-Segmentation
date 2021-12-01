#imports and libraries
import cv2
import numpy as np
from keras.models import load_model
from numpy.lib.arraypad import pad
from keras.preprocessing.image import img_to_array
from preprocess import characterSegment
import preprocess as p


# Line Segmentation
def lineSegment(image):
    #remove noise 
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15) 
    
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   
    # Code for enhancing the image--------------------------------------------------
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    
    #dilation
    kernel = np.ones((5,120), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   
    for i, ctr in enumerate(ctrs):

        # Get bounding box

        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y:y+h, x:x+w]

        # Pass image to wordSegment function
        wordSegment(roi)

        #add next line in arr_out
        p.arr_out.append("\n")
      

    final =" "
    # Get the value from arr_out and assign to String final
    for ch in (p.arr_out):
        i += 1
        final = final+ch
    return final
    

# WORD SEGMENTATION
def wordSegment(roi):  

    #remove noise 
    image1 = cv2.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 15) 

    #grayscale
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

    # Code for enhancing the image--------------------------------------------------
    #binary
    ret,thresh1 = cv2.threshold(gray1,127,255,cv2.THRESH_BINARY_INV)

    #dilation
    kernel1 = np.ones((10,50), np.uint8)
    img_dilation1 = cv2.dilate(thresh1, kernel1, iterations=1)

    #find contours
    ctrs1, hier = cv2.findContours(img_dilation1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs1 = sorted(ctrs1, key=lambda ctr1: cv2.boundingRect(ctr1)[0])
    
    for i, ctr1 in enumerate(sorted_ctrs1):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr1)

        # Getting ROI
        roi1 = roi[y:y+h, x:x+w]

        # Pass image to characterSegment function
        characterSegment(roi1)

        # show ROI
        #cv2.imshow('segment no:'+str(i),roi1)
        cv2.rectangle(roi,(x,y),( x + w, y + h ),(90,0,255),2)
        cv2.waitKey(0)


