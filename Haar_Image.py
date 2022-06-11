#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time


# In[2]:


#face_cascade = cv2.CascadeClassifier(r"C:\Users\vigna\OneDrive\Desktop\8f51e58ac0813cb695f3733926c77f52-07eed8d5486b1abff88d7e34891f1326a9b6a6f5\haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier(r"C:\Users\vigna\OneDrive\Desktop\8f51e58ac0813cb695f3733926c77f52-07eed8d5486b1abff88d7e34891f1326a9b6a6f5\haarcascade_frontalface_default.xml")
#nose_cascade = cv2.CascadeClassifier(r"C:\Users\vigna\OneDrive\Desktop\8f51e58ac0813cb695f3733926c77f52-07eed8d5486b1abff88d7e34891f1326a9b6a6f5\haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(r"C:\Users\vigna\OneDrive\Desktop\haarcascade_mcs_mouth.xml")
frame = cv2.imread(r"C:\Users\vigna\Downloads\WhatsApp Image 2022-06-05 at 6.07.41 PM.jpeg")
gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
haar_cascade = cv2.CascadeClassifier(r"C:\Users\vigna\OneDrive\Desktop\haarcascade_mcs_mouth.xml")


# In[3]:


faces = haar_cascade.detectMultiScale(gray_img,scaleFactor=1.5, minNeighbors=5)


# In[4]:


print(faces)


# In[5]:


for x,y,w,h in faces :
    image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    image = cv2.putText(frame,"Mouth",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,0),2)

    roi_gray = gray_img[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    #eyes = eye_cascade.detectMultiScale(roi_gray,1.3,3)

    #for ex,ey,ew,eh in eyes :
       # image = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        #image = cv2.putText(roi_color,"Eye",(ex,ey),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,0,0),2)


    # For Nose

    #nose =  nose_cascade.detectMultiScale(roi_gray,1.3,3)

    #for nx,ny,nw,nh in nose :
       # image = cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)
        #image = cv2.putText(roi_color,"Nose",(nx,ny),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,0,255),2)

    # For Mouth

    smile =  smile_cascade.detectMultiScale(roi_gray,1.9,25)
    
    for sx,sy,sw,sh in smile :
        image = cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
        image = cv2.putText(roi_color,"Mouth",(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,0),2)


img_resize = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
cv2.imshow("Frame",img_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




