from ultralytics import YOLO


class MLBirdClassifier:
    def __init__(
        self,
        model_path="/home/dys/pi/intelligent_bird_scarer/models/my_first_trained_model_ncnn_model",
    ):
        # Load the model once to save memory
        self.model = YOLO(model_path, task="detect")
        self.duck_id = 0  # ID for 'duck'
        self.target_id = [
            1,
            2,
            3,
            4,
            5,
        ]  # IDs for 'pheasant, pigeon, raven, sparrow, starling'

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
