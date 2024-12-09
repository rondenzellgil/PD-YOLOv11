if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("best-v4.pt")

    results = model.predict(source="test-vid-1.mp4", show=True, save=True, conf=0.7)
