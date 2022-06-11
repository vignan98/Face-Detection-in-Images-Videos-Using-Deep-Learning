#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2,time
import numpy as np

smile_cascade = cv2.CascadeClassifier(r"C:\Users\vigna\OneDrive\Desktop\haarcascade_mcs_mouth.xml")

video = cv2.VideoCapture(r"C:\Users\vigna\OneDrive\Desktop\Video.mp4")
n = 0
while (video.isOpened()):
    ret,frame = video.read()
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = smile_cascade.detectMultiScale(gray_img,scaleFactor=2.5, minNeighbors=7)
    for x,y,w,h in faces :
        image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        image = cv2.putText(frame,"Mouth",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,0),2)

        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # For Mouth

        smile =  smile_cascade.detectMultiScale(roi_gray,2.5,7)
        print(smile)
        for sx,sy,sw,sh in smile :
            image = cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
            image = cv2.putText(roi_color,"Mouth",(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,0),2)


    img_resize = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
    cv2.imshow("Frame",img_resize)

    key = cv2.waitKey(1)

    if key is ord('q'):
        break

video.release()
cv2.destroyAllWindows()


# In[ ]:




