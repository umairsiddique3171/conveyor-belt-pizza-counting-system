import pickle
import numpy as np
import cv2



def load_model(model_path): 
    model = pickle.load(open(model_path, "rb"))
    return model


def detect(img,model,scaler):
    img_resized = cv2.resize(img, (40, 20))
    img_gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
    flattened_img = np.reshape(img_gray, (1,800))
    flat_data = np.array(flattened_img)
    scaled_data = scaler.transform(flat_data)
    y_output = model.predict(scaled_data)
    print(y_output[0])
    if y_output[0] == 1:
        return True
    return False

