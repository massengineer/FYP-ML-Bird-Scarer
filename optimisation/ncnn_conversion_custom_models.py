from ultralytics import YOLO

# Use the full path to where 'best.pt' or 'my_[number]_trained_model.pt' is saved
model = YOLO("/home/dys/pi/intelligent_bird_scarer/models/my_fifth_trained_model.pt")

# 2. Export to NCNN
# imgsz: Should match the size used during training (usually 640)
# half:  Uses FP16 precision (highly recommended for Pi 5 to boost FPS)
# int8:  Optional, but half=True is usually the "sweet spot" for Pi 5
model.export(format="ncnn", imgsz=640, int8=True)
