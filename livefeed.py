if __name__ == "__main__":
    from ultralytics import YOLO
    import cv2

    model = YOLO("best-v4.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from the camera.")
                break

            results = model.predict(source=frame, show=True, conf=0.8)

            for result in results:
                annotated_frame = result.plot()
                cv2.imshow("YOLO Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
