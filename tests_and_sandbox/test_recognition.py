import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from libcamera import controls
from collections import deque

# 1. INITIALIZE VARIABLES OUTSIDE THE LOOP
# These must stay outside so they accumulate data over time
inference_history = deque(maxlen=100)
frame_count = 0
avg_inference = 0
avg_fps = 0

# Set up the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Load YOLOv8
model = YOLO("/home/dys/pi/FYP-ML-Bird-Scarer/models/my_fifth_trained_model_ncnn_model")

print("System Active. Press 'q' or 'Ctrl+C' to stop and see results.")

try:
    while True:
        frame = picam2.capture_array()
        results = model(frame)
        frame_count += 1

        # Current frame metrics
        inference_time = results[0].speed["inference"]
        fps = 1000 / inference_time if inference_time > 0 else 0

        # Update history
        inference_history.append(inference_time)

        # 2. CALCULATE STABLE AVERAGES
        if frame_count > 50:
            avg_inference = sum(inference_history) / len(inference_history)
            avg_fps = 1000 / avg_inference
            status_text = "Stable"
            color = (0, 255, 0)  # Green for stable
        else:
            avg_inference = inference_time
            avg_fps = fps
            status_text = "Warming up..."
            color = (0, 165, 255)  # Orange for warming up

        annotated_frame = results[0].plot()

        # 3. FIXED FORMATTING FOR ONSCREEN TEXT
        line1 = f"CURR: {fps:.1f} FPS | {inference_time:.1f} ms"
        line2 = f"AVG:  {avg_fps:.1f} FPS | {avg_inference:.1f} ms"
        line3 = f"STATUS: {status_text}"

        # Draw lines with consistent spacing on the left side for better readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            annotated_frame, line1, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(annotated_frame, line2, (20, 90), font, 1, color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, line3, (20, 130), font, 0.7, color, 1, cv2.LINE_AA)

        cv2.imshow("Smart Bird Scarer - Telemetry", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

# 4. EXCEPTION HANDLING FOR KEYBOARD INTERRUPT
except KeyboardInterrupt:
    print("\n[INFO] Keyboard Interrupt detected. Calculating final results...")

# 5. FINAL PRINTING (Runs whether you press 'q' or 'Ctrl+C')
finally:
    if frame_count > 0:
        print("\n" + "=" * 30)
        print("   FINAL ENGINEERING METRICS")
        print("=" * 30)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Final Average Inference: {avg_inference:.2f} ms")
        print(f"Final Average FPS:       {avg_fps:.2f}")
        print("=" * 30)

    picam2.stop()
    cv2.destroyAllWindows()
