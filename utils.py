import pickle
import numpy as np
import cv2


def load_model(model_path): 
    model = pickle.load(open(model_path, "rb"))
    return model


def detect(img, model, scaler, threshold=0.7):
    img_resized = cv2.resize(img, (7, 4))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    flattened_img = np.reshape(img_rgb, (1, 84))
    flat_data = np.array(flattened_img)
    scaled_data = scaler.transform(flat_data)
    y_prob = model.predict_proba(scaled_data)
    if y_prob[0][1] > threshold:
        return True # pizza 
    return False # no_pizza

