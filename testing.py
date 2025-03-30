if __name__ == "__main__":
    from ultralytics import YOLO
    import cv2

    model = YOLO(r"C:\Users\ronde\Desktop\PD\yolov11-v7\runs\detect\train4\weights\best.pt")

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

            results = model.predict(source=frame, conf=0.5, verbose=False)  # Removed show=True for manual display

            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Real-time frame processing
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
