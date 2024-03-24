import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import sklearn 
import json
import numpy as np
from utils import load_model, detect


# load model
model_path = r"C:\Users\US593\OneDrive\Desktop\conveyor_belt_pizza_counter\classifier\naive_bayes_model.p"
model = load_model(model_path)

# load scaler (for normalization)
scaler_path = r"C:\Users\US593\OneDrive\Desktop\conveyor_belt_pizza_counter\classifier\min_max_scaler.p"
scaler = load_model(scaler_path)

# region of interest (roi) [x1,y1,x2,y2]
# roi = [[378,101,397,103],[385,105,401,108],[527,232,557,238],[539,246,575,252]]
roi = [[378,101,385,105],[385,106,392,110],[528,235,535,239],[539,251,546,255]]

# load frames
cap = cv2.VideoCapture('video.mp4')

# read frames
ret = True
frame_number = 0
count = 0
to_count = False

while ret:

    ret, frame = cap.read()
    frame_number += 1
    if cv2.waitKey(25) & 0xFF == ord('n'):
        break
    
    if frame_number%5==0 or frame_number==1:
        do_detection = True
    for x1,y1,x2,y2 in roi:
        if do_detection:
            detection = detect(frame[y1:y2,x1:x2],model,scaler)
            prev_detection = detection
        else:
            detection = prev_detection
        print(detection)
        if detection:
            count +=1  
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2), int(y2)), (0, 255,0), 1)
            to_count = True
        else: 
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2), int(y2)), (0,0,255), 1)

    # write text 
    cv2.putText(frame, 'Counts : {}'.format(str(count)), (20,27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # show results
    cv2.imshow('frame', frame)

print(f'Total Counts : {count}')
cap.release()
cv2.destroyAllWindows()