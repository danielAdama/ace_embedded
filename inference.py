import cv2
import datetime
from config import config
from vision.vision_tracker import VisionTracker
import logging
logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.NullHandler())

webcam = cv2.VideoCapture(config.VIDEO)
frame_count = 0
vision_tracker = VisionTracker(config)

while True:
    success, frame = webcam.read()
    if not success:
        break

    frame_count += 1
    detections, detected_ids, track_id_list, tracks = vision_tracker.process_frame(frame, frame_count)
    detected_object_names = [detections.names[class_id] for class_id in detected_ids]
    for track in tracks:
        if not track.is_confirmed():
            continue

        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = map(int, ltrb)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), config.GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), config.GREEN, -1)
        cv2.putText(frame, str(track.track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.WHITE, 2)

    logging.info("------------")
    logging.info("Time(YYYY-MM-DD HH:MM:SS.ssssss):", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    logging.info("Detected IDs:", )
    logging.info("Object Track IDs:", track_id_list)
    logging.info("Frame Size (height, width in pixels):", frame.shape[:2])
    
    cv2.imshow('Live', frame)
    key = cv2.waitKey(5)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()