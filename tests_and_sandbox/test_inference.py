from ultralytics import YOLO

# Load model (Explicitly define task to remove the Warning)
model = YOLO(
    "/home/dys/pi/FYP-ML-Bird-Scarer/models/my_fifth_trained_model_ncnn_model",
    task="detect",
)

# Use an image file instead of the camera (source=0)
# Use one of your validation images to get a real-world result
results = model.predict(
    source="/home/dys/pi/FYP-ML-Bird-Scarer/fifth_model_validation_images/images/0be64d0484da7f5d_jpg.rf.04b6d1586df6ac3ba64640923855f992.jpg",
    imgsz=640,
    half=True,
)

# Extract timing in milliseconds
inference_ms = results[0].speed["inference"]
preprocess_ms = results[0].speed["preprocess"]
postprocess_ms = results[0].speed["postprocess"]

total_latency = inference_ms + preprocess_ms + postprocess_ms
print(f"Total Latency: {total_latency:.2f} ms")
print(f"Inference only: {inference_ms:.2f} ms")
print(f"Preprocess: {preprocess_ms:.2f} ms")
print(f"Postprocess: {postprocess_ms:.2f} ms")
