if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO(r"C:\Users\ronde\Desktop\PD\yolov11-v7\yolov11-v7\runs\detect\train4\weights\best.pt")

    results = model.predict(source="test\images", show=True, save=True, conf=0.7)
