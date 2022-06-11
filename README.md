# Face-Detection-in-Images-Videos-Using-Deep-Learning

I have been working on a project "Face Detection in Images/Videos". I particularly focused on detecting mouth region to track the position of the mouth during speech. Being completely new to this domain, I have explored different techniques following some blogs, Research papers. I focused on Images first and then applied the same on Videos

Few techniques that I have implemented:

1.Face Detection using Haar-cascade features(Viola-Jones Method) is the basic approach to start with. In this technique Haar-features are applied on an image to find out the edges, lines, difference in color regions of the image. Based on the difference in pixel values between the darker & lighter regions features are detected.


Key points that Viola-Jones has introduced in its Research Paper:

1. Integral Image - Makes computation easy
2. Adaboost algorithm - Feature Selection
3.Combining most efficient classifiers in cascade

Drawbacks with this approach:

This algorithm is more prone to False Positives (Detecting the non-facial region as face - you can observe it from the below image). Tuning the parameters 'scaleFactor'( determines by what factor the image size reduces at each stage) 'minNeighbors'(how many neighbors each candidate rectangle should have in order to be considered as a face) made some accurate predictions but resulted in False Negatives. Changing these parameter values manually from image to image to find the best pair is difficult.

Advantage with this approach: This technique is faster compared to other existing techniques.


2. To overcome these problems Multi-task cascade deep learning neural network(MTCNN's) has been developed which uses convolutional neural networks to detect the faces. This model uses three convolutional networks (P-Net, R-Net, O-Net). P-Net uses bounding box regression vectors & NMS (Non-Maximum Suppression) to detect the faces & to select the appropriate bounding boxes from the one’s that overlap. R-Net further eliminates the unimportant bounding boxes with low confidence levels & through applying NMS and finally passes it to O-Net. The output of O-Net are the co-ordinates of bounding box( The Face detection), 5 facial landmarks(I have used three in my code to focus on mouth region). This technique infact reduced false predictions and made some accurate face detections( The image with blue marks)


I tried the same with video input & it worked well. I was able to track the mouth region of the speaker in a video for the whole duration.

Image1: Haar Cascade Technique (scaleFactor=1.5, minNeighbors=5) ,2 False Negatives, 0 FP.

Image2: Haar Cascade Technique (scaleFactor=2,minNeighbors=2) , 7 False Positives, 0 FN.

Image3 : Using MTCNN Accurate detection of faces was achieved ( The one in blue marks)
