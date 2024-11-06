import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

# Define paths
image_directory = "C:\\Users\\faraz\\PycharmProjects\\Face-AntiSpoofing\\data"
real_folder = os.path.join(image_directory, "real")
spoof_folder = os.path.join(image_directory, "spoof")

# Initialize face detector and anti-spoof model
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')


def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]

    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)

    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
    y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)

    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img,
                             y1 - y, int(l * bbox_inc - y2 + y),
                             x1 - x, int(l * bbox_inc) - x2 + x,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

# Define function to make a prediction
def make_prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img])[0]

    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None, None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)

    return label, score


# Read images and make predictions
y_true = []
y_pred = []

for label, folder in enumerate([real_folder, spoof_folder]):
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Failed to read {file_path}")
            continue

        predicted_label, score = make_prediction(img)
        if predicted_label is not None:
            y_true.append(label)  # 0 for real, 1 for spoof
            y_pred.append(predicted_label)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
