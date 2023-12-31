import cv2
import datetime
from config import config
from vision.vision_tracker import VisionTracker
import logging
logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.NullHandler())


cap = cv2.VideoCapture(config.VIDEO)
frame_count = 0
vision_tracker = VisionTracker(config)

if not cap.isOpened():
    raise ValueError("Could not open video stream")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (height, width) = frame.shape[:2]
    predicted_caption = vision_tracker.generate_caption(frame_rgb)
    results, detections, detected_ids, track_id_list, tracks = vision_tracker.process_frame(frame_rgb, frame_count)
    detected_object_names = [detections.names[class_id] for class_id in detected_ids]
    obj_dim = [[box[0][0], box[0][1], box[0][2], box[0][3]] for box in results]
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
    logging.info("Time(YYYY-MM-DD HH:MM:SS.ssssss): %s", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    logging.info("Objects: %s", detected_object_names)
    logging.info("Object Bounding Boxes (x, y, w, h): %s", obj_dim)
    logging.info("Object Track IDs: %s", track_id_list)
    logging.info("Frame Size (height, width in pixels): (%d, %d)", height, width)
    logging.info("Frame Summary: %s", predicted_caption.capitalize())
    
    cv2.imshow('Live', frame)
    key = cv2.waitKey(5)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()