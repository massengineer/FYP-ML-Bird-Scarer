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
        """Processes a frame and returns a 3-tuple:
        (is_bird: bool, annotated_frame: np.ndarray, target_flags: list[bool])

        `target_flags` mirrors `self.target_id` (e.g. [raven_found, sparrow_found]).
        """
        # Default outputs
        is_bird = False
        # If model.plot isn't available because there are no results, return the raw frame
        annotated_frame = frame
        target_flags = [False] * len(self.target_id)

        results = self.model(frame, stream=True, verbose=False)

        for result in results:
            # Create the frame with boxes/labels for display or debugging
            try:
                annotated_frame = result.plot()
            except Exception:
                # If plotting fails, keep original frame
                pass

            # Check if any detected box is a pest bird with > 80% confidence
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id in self.target_id and conf > 0.8:
                    is_bird = True
                    # Mark which target was found
                    try:
                        idx = self.target_id.index(cls_id)
                        target_flags[idx] = True
                    except ValueError:
                        # Shouldn't happen, but ignore if it does
                        pass
                    print(f"Detected: {result.names[cls_id]} ({conf:.2f})")

        return is_bird, annotated_frame, target_flags
