if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolo11s.pt")

    results = model.train(data="dataset-v4.yaml", patience=10, device=0, plots=True)
