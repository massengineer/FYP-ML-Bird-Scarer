import cv2
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


while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Run YOLO model on the captured frame and store the results
    results = model(frame)

    # Output the visual detection data, we will draw this on our camera preview window
    annotated_frame = results[0].plot()

    # Get inference time
    inference_time = results[0].speed["inference"]
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f"FPS: {fps:.1f}"

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the text on the annotated frame
    cv2.putText(
        annotated_frame,
        text,
        (text_x, text_y),
        font,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
