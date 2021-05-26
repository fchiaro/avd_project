import json
import os
import sys
import cv2
import math

sys.path.append(os.path.abspath('./traffic_light_detection_module/')) # the append is temporary, just for the execution time of this module

from yolo import YOLO
from postprocessing import draw_boxes


# detector_window = cv2.namedWindow(DETECTOR_STATE_WINDOW_NAME, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(DETECTOR_STATE_WINDOW_NAME, camera_parameters['width'],camera_parameters['height'])

class TrafficLightDetector:
    def __init__(self, config_path, show_detections=True):
        self._show_detections = show_detections
        self._DETECTOR_STATE_WINDOW_NAME = 'Detector state'

        with open(os.path.abspath(config_path), 'r') as detector_config_file:
            self._config = json.load(detector_config_file) # save the configuration since it contains info about network's input size and so on

        self._detector = YOLO(config=self._config)

    def detect_and_estimate_distance(self, bgr_image, depth_image_array):
        """[summary]

        Args:
            bgr_image ([type]): [description]
            depth_image_array ([type]): the normalized array obtained by the depth image (already multiplied by 1000). 

        Returns:
            [type]: [description]
        """
        semaphore_predictions = self._detector.predict(bgr_image)

        if self._show_detections:
            img_resized = cv2.resize(bgr_image, dsize=(self._config['model']['image_w'], self._config['model']['image_h']))
            plt_image = draw_boxes(img_resized, semaphore_predictions, self._config['model']['classes'])
            cv2.imshow(self._DETECTOR_STATE_WINDOW_NAME, plt_image)
            cv2.waitKey(1)
        
        if len(semaphore_predictions) == 0:
            # no semaphore has been detected
            return None, None
        
        # Detector is able to find mutiple semaphores in the same image. We consider only the one with the highest score because it's less likely
        # to be a false positive (we are assuming an ODD involving only one semaphore per lane).

        # we know that at least a prediction exists (otherwise we would have returned above)
        highest_confidence_prediction = semaphore_predictions[0]
        for i in range(1, len(semaphore_predictions)):
            if semaphore_predictions[i].get_score() > highest_confidence_prediction.get_score():
                highest_confidence_prediction = semaphore_predictions[i]
        
        semaphore_distance = self._estimate_distance(highest_confidence_prediction, depth_image_array)

        return highest_confidence_prediction.get_label(), semaphore_distance

    def _estimate_distance(self, semaphore_prediction, depth_image_array):
        """[summary]

        Args:
            bgr_image ([type]): [description]
            semaphore_prediction ([type]): [description]
            depth_image_array ([type]): the normalized array obtained by the depth image (already multiplied by 1000). 
        """
        # offset w.r.t. RGB camera (can't physically put two cameras in the exact same place)
        # offset of depth camer w.r.t. car's front (bonnet)
        # once we have considered these two offset, we have the distance between the farthest front point of the car
        # and the semaphore
        # we ignore these offset because of presence of largers error due to camera noise and detector errors

        depth_data = depth_image_array[math.floor(semaphore_prediction.ymin*depth_image_array.shape[0]):math.floor(semaphore_prediction.ymax*depth_image_array.shape[0]), \
            math.floor(semaphore_prediction.xmin*depth_image_array.shape[1]):math.floor(semaphore_prediction.xmax*depth_image_array.shape[1])]
        
        return depth_data.mean()
