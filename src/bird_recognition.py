from ultralytics import YOLO


class MLBirdClassifier:
    def __init__(self, model_path="yolov8n_ncnn_model"):
        # Load the model once to save memory
        self.model = YOLO(model_path, task="detect")
        self.target_id = 14  # COCO ID for 'bird'

    def scan_frame(self, frame):
        """Processes a frame and returns (is_bird, annotated_image)"""
        results = self.model(frame, stream=True, verbose=False)
        is_bird = False

        for result in results:
            # Check if any detected box is a bird with > 50% confidence
            for box in result.boxes:
                if int(box.cls[0]) == self.target_id and float(box.conf[0]) > 0.5:
                    is_bird = True

            # Create the frame with boxes/labels for display or debugging
            annotated_frame = result.plot()
            return is_bird, annotated_frame

        return False, frame
