if __name__ == "__main__":
    from ultralytics import YOLO
    import cv2
    from gpiozero import Servo
    from time import sleep

    # Initialize the YOLO model
    model = YOLO("best-v4.pt")  # Your trained model for vials/ampoules

    # Initialize the servo (connected to GPIO 18)
    servo = Servo(18)

    # Servo positions (adjust if needed)
    SEGREGATE_POS = -1  # Move to segregate position
    KEEP_POS = 1        # Stay in position (do not segregate)

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

            results = model.predict(source=frame, show=True)

            detected_vial_ampoule = False  # Flag to check if we detect a vial/ampoule

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)  # Get class index
                    class_name = model.names[class_id]  # Get class name
                    print(f"Detected: {class_name}")

                    if class_name in ["vial", "ampoule"]:  # Check if it's a vial or ampoule
                        detected_vial_ampoule = True

            # Control the servo based on detection
            if detected_vial_ampoule:
                print("✅ Vial/Ampoule detected: DO NOT segregate")
                servo.value = KEEP_POS  # Keep position
            else:
                print("⚠️ No Vial/Ampoule detected: SEGREGATE")
                servo.value = SEGREGATE_POS  # Move to segregate position

            sleep(0.5)  # Small delay to allow movement

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
