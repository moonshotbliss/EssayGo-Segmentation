#imports and libraries
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



arr_out = []
arr_result = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','l','M','N','O','P','Q','R','S','T','u','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']


#load model to variable model
model=load_model('model/model.h5')


#Prediction of character present in the region of interest
def predict(x,y,w,h,imd):
    # Getting ROI                
    test=imd[y:y+h, x:x+w]

    # Code for enhancing the image--------------------------------------------------
    _,test_image = cv2.threshold(test,100,255,cv2.THRESH_BINARY)
    test_image= cv2.copyMakeBorder(test_image,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
    test_image = cv2.medianBlur(test_image.copy(),3)

    #resize
    test_image = cv2.resize(test_image.copy(),(28,28),interpolation = cv2.INTER_AREA)
    
    #preprocess
    test_image=(image.img_to_array(test_image))/255
    test_image=np.expand_dims(test_image, axis = 0)

    #model predict
    result=model.predict(test_image).round()

    #reshape according to number of classes
    np.reshape(result, 47)

    
    
    high = np.amax(test_image)
    low = np.amin(test_image)

    #get the highest value from result
    if high != low:
        maxval = np.amax(result)
        index = np.where(result == maxval)
        arr_out.append(arr_result[index[1][0]])


#Character Segmentation 
def characterSegment(input_img):  

    #remove noise 
    im = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 15) 

    #grayscale
    img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    
    # Code for enhancing the image--------------------------------------------------
    #binary
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
   
    #dilation
    kernel = np.ones((10,1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
   
    #find contours
    contours, h = cv2.findContours(img_dilation.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    contours = sorted(contours, key=lambda ctr2: cv2.boundingRect(ctr2)[0])

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        
        #call predict function
        predict(x,y,w,h,img)

        #plot the rectange in characters   
        cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)


    final = ""
    # add space in arr_out
    arr_out.append(" ")
    
