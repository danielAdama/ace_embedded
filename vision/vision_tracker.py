from deep_sort_realtime.deepsort_tracker import DeepSort
from ordered_set import OrderedSet

class VisionTracker:
    def __init__(self, config):
        self.config = config
        self.tracker = DeepSort(
            max_age=config.MAX_AGE,
            max_cosine_distance=config.MAX_COS_DIST,
            nn_budget=config.NN_BUDGET,
            embedder=config.EMBEDDER,
            embedder_gpu=config.is_GPU,
            embedder_model_name=config.EMBEDDER_MODEL
        )
        self.track_id_set = OrderedSet()
        self.detected_ids = []
        self.prev_frame_list = []
        self.cur_frame_set = set()

    def process_frame(self, frame, frame_count):
        self.frame = frame
        self.frame_count = frame_count
        detections = self.config.MODEL(self.frame)[0]
        results = []

        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < self.config.CONFI_THRESH:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        self.prev_frame_list = list(self.cur_frame_set).copy()
        track_id_list = list(sorted(self.track_id_set))
        tracks = self.tracker.update_tracks(results, frame=self.frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            self.detected_ids.clear()
            for class_ids in results:
                for class_id in class_ids:
                    if class_id in self.config.ALLOWED_IDS:
                        track_id = track.track_id
                        self.cur_frame_set.add(self.frame_count)
                        if len(self.cur_frame_set) > len(self.prev_frame_list):
                            self.detected_ids.append(class_id)
                            self.track_id_set.add(track_id)

        return detections, self.detected_ids, track_id_list, tracks