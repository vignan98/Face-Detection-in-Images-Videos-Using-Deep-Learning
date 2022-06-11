#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
from mtcnn.mtcnn import MTCNN


# In[11]:


detector = MTCNN()
cap = cv2.VideoCapture(r"C:\Users\vigna\OneDrive\Desktop\Video.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            values1=keypoints['mouth_left']
            values2=keypoints['mouth_right']
            #print(values1)
            #print(values2)
            cv2.rectangle(frame,
                          (values1[0], values1[1]),(values2[0],values2[1]),
                          (0,155,255),
                          2)
    out.write(frame)        
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:




