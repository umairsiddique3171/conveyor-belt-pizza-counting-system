import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# importing libraries
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
roi = [[397,137,404,144],[417,159,424,166]]

# load frames
cap = cv2.VideoCapture('pizza_line.mp4')

# # for saving results
ret,frame = cap.read()
output_video = cv2.VideoWriter(os.path.join('.','results','results.mp4'),
                              cv2.VideoWriter_fourcc(*'MP4V'),
                              25,
                              (frame.shape[1],frame.shape[0]))

# read frames
ret = True
frame_number = 0
count = 0
prev_count = [False]*len(roi)
while ret:

    ret, frame = cap.read()
    frame_number += 1
    if cv2.waitKey(25) & 0xFF == ord('n'):
        break
    
    spot = 0
    # spot detection
    for x1, y1, x2, y2 in roi:
        if frame_number == 1 or frame_number % 5 == 0:
            detection = detect(frame[y1:y2, x1:x2], model, scaler)

        if detection: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            to_count = True  
        else: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            to_count = False
        
        # spot count
        if to_count:
            if spot == 0 and not prev_count[spot]:
                prev_count[spot] = True
                count+=1
            elif spot == 1 and not prev_count[spot]:
                prev_count[spot] = True
                count += 1
        else:
            if spot == 0:
                prev_count[spot] = False
            elif spot == 1:
                prev_count[spot] = False
        spot += 1

    # write text 
    cv2.putText(frame, 'Counts : {}'.format(str(count)), (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show results
    if ret: 
        cv2.imshow('frame', frame)

    # writing output to results/results.mp4
    output_video.write(frame)

print(f'Total Counts : {count}')
cap.release()
cv2.destroyAllWindows()