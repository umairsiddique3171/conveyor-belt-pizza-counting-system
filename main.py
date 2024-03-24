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
roi = [[378,101,385,105],[385,106,392,110],[528,235,535,239],[539,251,546,255]]

# load frames
cap = cv2.VideoCapture('video.mp4')

# read frames
ret = True
frame_number = 0
count = 0
prev_count1,prev_count2,prev_count3,prev_count4 = False,False,False,False
while ret:

    ret, frame = cap.read()
    frame_number += 1
    if cv2.waitKey(25) & 0xFF == ord('n'):
        break
    
    do_detection = frame_number == 1 or frame_number % 7 == 0  # Set do_detection to True every 5 frames or in the first frame
    spot = 0

    for x1, y1, x2, y2 in roi:
        spot += 1
        if do_detection:
            detection = detect(frame[y1:y2, x1:x2], model, scaler)

        if detection: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            to_count = True
          
        else: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            to_count = False

        if to_count:
            if spot == 1 and not prev_count1:
                prev_count1 = True
                count += 1
                cv2.imwrite(f"data/{frame_number}_{spot}.jpg",frame[y1:y2,x1:x2])
            elif spot == 2 and not prev_count2:
                prev_count2 = True
                count += 1
            elif spot == 3 and not prev_count3:
                prev_count3 = True
                count += 1
            elif spot == 4 and not prev_count4:
                prev_count4 = True
                count += 1
        else:
            if spot == 1:
                prev_count1 = False
            elif spot == 2:
                prev_count2 = False
            elif spot == 3:
                prev_count3 = False
            elif spot == 4:
                prev_count4 = False

    
    # write text 
    cv2.putText(frame, 'Counts : {}'.format(str(count)), (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # show results
    cv2.imshow('frame', frame)

print(f'Total Counts : {count}')
cap.release()
cv2.destroyAllWindows()
