#!/usr/bin/env python
# coding: utf-8

# In[3]:


from mtcnn import MTCNN
import cv2
image_path = r"C:\Users\vigna\Downloads\WhatsApp Image 2022-06-05 at 6.07.41 PM.jpeg"
img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
detector = MTCNN()
detections = detector.detect_faces(img)
detections


# In[6]:


import matplotlib.pyplot as plt

img_with_dets = img.copy()
min_conf = 0.9
for det in detections:
    if det['confidence'] >= min_conf:
        x, y, width, height = det['box']
        keypoints = det['keypoints']
        values1=keypoints['mouth_left']
        values2=keypoints['mouth_right']
        cv2.rectangle(img_with_dets,
                         (values1[0], values1[1]),(values2[0],values2[1]),
                          (0,155,255),
                          2)
        #cv2.rectangle(img_with_dets, (x,y), (x+width,y+height), (0,155,255), 2)
        #cv2.circle(img_with_dets, (keypoints['left_eye']), 2, (0,155,255), 2)
        #cv2.circle(img_with_dets, (keypoints['right_eye']), 2, (0,155,255), 2)
        #cv2.circle(img_with_dets, (keypoints['nose']), 2, (0,155,255), 2)
        #cv2.circle(img_with_dets,(keypoints['mouth_left']), 2,  (0,155,255), 2)
        #cv2.circle(img_with_dets,(keypoints['mouth_right']),2, (0,155,255), 2)
plt.figure(figsize = (10,10))
plt.imshow(img_with_dets)
plt.axis('off')


# In[ ]:




