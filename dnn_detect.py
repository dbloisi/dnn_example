import cv2 as cv
import numpy as np

# load classes available in the network
classes = []
with open('classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#load network data
cvNet = cv.dnn.readNetFromTensorflow(
               'frozen_inference_graph.pb',
               'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

#load an image to test the network
img = cv.imread('table.jpg')
img_height = img.shape[0]
img_width = img.shape[1]

#feed the image into the network
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

#feed forward the model
cvOut = cvNet.forward()

#thresholding results
for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        #prediction class index
        idx = int(detection[1])        
        #drawing the bounding box
        left = detection[3] * img_width
        top = detection[4] * img_height
        right = detection[5] * img_width
        bottom = detection[6] * img_height
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), colors[idx], thickness=2)
        #converting id to class name   
        label = "{}: {:.2f}".format(classes[idx],score)
        y = top - 5 if top - 5 > 5 else top + 5
	#adding text (with background)
        cv.rectangle(img, (int(left), int(y-15)), (int(left + 150), int(top)), colors[idx], thickness=-1)
        cv.putText(img, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

#showing results
cv.imshow('detection', img)
cv.waitKey(0)
cv.destroyAllWindows()

