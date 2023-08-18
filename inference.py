import datetime
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import config
from ordered_set import OrderedSet
import logging
logging.basicConfig(filename='output_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.NullHandler())


model = YOLO("yolov8s.pt")
tracker = DeepSort(
    max_age=50,
    max_cosine_distance=0.3,
    nn_budget=5,
    embedder="mobilenet",
    embedder_gpu=True,
    embedder_model_name="mars-small128.pb",
)
s
ALLOWED_IDS = [0, 2]
track_id_set = OrderedSet()
detected_ids = []
prev_frame_list = []
cur_frame_set = set()
frame_count = 0

webcam = cv2.VideoCapture(config.VIDEO)
CONFI_THRESH = 0.5

while True:
    success, frame = webcam.read()
    if not success:
        break
    frame_count += 1
    (height, width) = frame.shape[:2]
    detections = model(frame)[0]
    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < CONFI_THRESH:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    prev_frame_list = list(cur_frame_set).copy()
    track_id_list = list(sorted(track_id_set))
    tracks = tracker.update_tracks(results, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue

        detected_ids.clear()
        for class_ids in results:
            for class_id in class_ids:
                if class_id in ALLOWED_IDS:
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    cur_frame_set.add(frame_count)

                    if len(cur_frame_set) > len(prev_frame_list):
                        detected_ids.append(class_id)
                        track_id_set.add(track_id)

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    logging.info("Time(YYYY-MM-DD HH:MM:SS.ssssss): %s", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    logging.info("Objects: %s", [detections.names[class_id] for class_id in detected_ids])
    logging.info("Object Bounding Boxes (x, y, w, h): %s", [[box[0][0], box[0][1], box[0][2], box[0][3]] for box in results])
    logging.info("Object Track IDs: %s", track_id_list)
    logging.info("Frame Size (height, width in pixels): (%d, %d)", height, width)
    # logging.info("Frame Summary: )
    # print("Output has been logged to output_log.txt")
    

    cv2.imshow('Live', frame)
    key = cv2.waitKey(5)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()