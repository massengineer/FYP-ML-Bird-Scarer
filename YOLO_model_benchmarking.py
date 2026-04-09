import pandas as pd
import matplotlib.pyplot as plt

# YOLO model data
data = [
    ["YOLO26", "N", 1.7, 40.9],
    ["YOLO26", "S", 2.5, 48.6],
    ["YOLO26", "M", 4.7, 53.1],
    ["YOLO26", "L", 6.2, 55.0],
    ["YOLO26", "X", 11.8, 57.5],
    ["YOLO11", "N", 1.5, 39.5],
    ["YOLO11", "S", 2.5, 47.0],
    ["YOLO11", "M", 4.7, 51.5],
    ["YOLO11", "L", 6.2, 53.4],
    ["YOLO11", "X", 11.3, 54.7],
    ["YOLOv8", "N", 1.47, 37.3],
    ["YOLOv8", "S", 2.66, 44.9],
    ["YOLOv8", "M", 5.86, 50.2],
    ["YOLOv8", "L", 9.06, 52.9],
    ["YOLOv8", "X", 14.37, 53.9],
    ["YOLOv5", "N", 1.12, 28.0],
    ["YOLOv5", "S", 1.92, 37.4],
    ["YOLOv5", "M", 4.03, 45.4],
    ["YOLOv5", "L", 6.61, 49.0],
    ["YOLOv5", "X", 11.89, 50.7],
]

df = pd.DataFrame(data, columns=["model", "version", "latency", "map"])

version_order = ["N", "S", "M", "L", "X"]
df["version"] = pd.Categorical(df["version"], categories=version_order, ordered=True)

df = df.sort_values(["model", "version"])

plt.figure(figsize=(10, 6))

for model, g in df.groupby("model", sort=False):
    plt.plot(g["latency"], g["map"], marker="o", linewidth=2, markersize=7, label=model)

    for _, row in g.iterrows():
        plt.annotate(
            row["version"],
            (row["latency"], row["map"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

plt.xlabel("Latency T4 TensorRT10 FP16 (ms/img)")
plt.ylabel("COCO mAP 50-95")
plt.title("Model Family Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
