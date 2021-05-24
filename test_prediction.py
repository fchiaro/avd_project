import json
import cv2
import os
import sys

sys.path.append(os.path.abspath('./traffic_light_detection_module/')) # the append is temporary, just for the execution time of this module

from yolo import YOLO
from postprocessing import draw_boxes

CONFIG_PATH = 'C:/CarlaSimulator/PythonClient/avd_project/traffic_light_detection_module/config.json'
IMG_PATH = 'C:/CarlaSimulator/PythonClient/avd_project/traffic_light_detection_module/test_images/test (10).png'

with open(CONFIG_PATH, 'r') as json_file:
    config = json.load(json_file)

model = YOLO(config=config)

img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
cv2.imshow('Input image',img)
cv2.waitKey(2000)

predictions = model.predict(img)

plt_image = draw_boxes(img, predictions, config['model']['classes'])

cv2.imshow('Prediction', plt_image)
cv2.waitKey(10000)
