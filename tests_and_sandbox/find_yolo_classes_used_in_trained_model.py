from ultralytics import YOLO

model = YOLO(
    "/home/dys/pi/intelligent_bird_scarer/train_data_for_third_model/weights/best.pt"
)
print(model.names)  # This returns a dictionary: {0: 'class1', 1: 'class2', ...}
