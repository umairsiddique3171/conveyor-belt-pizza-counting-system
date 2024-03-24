if score > 0.65:
                detections_.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)