import cv2
from config import config
import os

class CarDetector():
    """Car Detector class which detects the presence of vehicles (cars) in an image utilizing YOLOv4.
    
    Attributes:
        
    """
    def __init__(self, weight = "yolov4-tiny.weights", cfg = "yolov4-tiny.cfg"):

        net = cv2.dnn.readNet(
            os.path.join(os.getcwd(),f"dnn_model/{weight}"),
            os.path.join(os.getcwd(),f"dnn_model/{cfg}"))
        
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(732, 732), scale=1/255.0)
        # 2-car, 5-bus, 7-van
        # Allow class containing Cars only
        self.classe_allowed = 2
    
    def vehicle_detected(self, frame):
        car_boxes = []
        class_ids, confidences, boxes = self.model.detect(frame, nmsThreshold=0.4)
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            # Skip detections with low confidence
            if (confidence < 0.86):
                continue
            if (class_id == self.classe_allowed):
                car_boxes.append(box)
        
        return car_boxes