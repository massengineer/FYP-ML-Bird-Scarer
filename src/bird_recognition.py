from ultralytics import YOLO


class MLBirdClassifier:
    def __init__(
        self,
        model_path="/home/dys/pi/FYP-ML-Bird-Scarer/models/my_fifth_trained_model_ncnn_model",
    ):
        # Load the model once to save memory
        self.model = YOLO(model_path, task="detect")
        self.target_id = [
            1,
            2,
        ]  # IDs for 'raven, sparrow' and ID for 'duck' is 0 but it is not a pest bird so it has not been added to the target_id list

    def scan_frame(self, frame):
        """Processes a frame and returns (is_bird, annotated_image)"""
        results = self.model(frame, stream=True, verbose=False)
        is_bird = False

        for result in results:
            # Check if any detected box is a pest bird with > 50% confidence
            for box in result.boxes:
                if int(box.cls[0]) in self.target_id and float(box.conf[0]) > 0.5:
                    is_bird = True
                    print(
                        f"Detected: {result.names[int(box.cls[0])]} ({float(box.conf[0]):.2f})"
                    )

            # Create the frame with boxes/labels for display or debugging
            annotated_frame = result.plot()

        return is_bird, annotated_frame
