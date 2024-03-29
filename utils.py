import pickle
import numpy as np
import cv2


def load_model(model_path): 
    model = pickle.load(open(model_path, "rb"))
    return model


def detect(img, model, scaler):
    img_resized = cv2.resize(img, (7,7))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    flattened_img = np.reshape(img_rgb, (1,147))
    flat_data = np.array(flattened_img)
    scaled_data = scaler.transform(flat_data)
    output = model.predict(scaled_data)[0]
    if output == 1:
        return True # pizza 
    return False # no_pizza

