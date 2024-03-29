# saving frame for coordinate selection for region of interest. 

import cv2

cap = cv2.VideoCapture('pizza_line.mp4')

ret = True
frame_number = 0

while ret : 
    ret,frame = cap.read()
    frame_number += 1
    if frame_number == 10:
        cv2.imwrite('frame.jpg',frame)
        break

cap.release()
cv2.destroyAllWindows()
