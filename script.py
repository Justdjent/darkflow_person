from darkflow.net.build import TFNet
import cv2
import tensorflow as tf
from time import time as timer
import sys
import os

input_file = sys.argv[1]

output_file = sys.argv[2]
#options of darkflow - model, weights
options = {"model": "cfg/yolo_new.cfg", "load": -1, "threshold": 0.65, "gpu":0.7}
#create a tfnet object
tfnet = TFNet(options)

#read video from file
cap = cv2.VideoCapture(input_file)
#cap = cv2.VideoCapture(0)

#save parameters of video
width=int(cap.get(3))

height=int(cap.get(4))

frame_rate=int(cap.get(5))

#initializing videocodec and video writer
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_file,fourcc, frame_rate, (width,height))

elapsed = int()
start = timer()
#reading video file an image by image while there is images
#to see images in realtimework you can uncomment string cv2.imshow('',preprocess)
while cap.isOpened():
    ret, frame = cap.read()
    if ret==False: break
    preprocessed = tfnet.framework.preprocess(frame)
    feed_dict = {tfnet.inp: [preprocessed]}
    net_out = tfnet.sess.run(tfnet.out,feed_dict)[0]
    processed = tfnet.framework.postprocess(net_out, frame, False)
    out.write(processed)
    #cv2.imshow('', processed)
    elapsed += 1
    if elapsed % 5 == 0:
        sys.stdout.write('\r')
        sys.stdout.write('{0:3.3f} FPS'.format(
            elapsed / (timer() - start)))
        sys.stdout.flush()
    choice = cv2.waitKey(1)
    if choice == 27: break
#writing video to file    
sys.stdout.write('\n')
cap.release()
out.release()
cv2.destroyAllWindows()