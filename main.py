from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Error: Webcam not found")
    exit()

print("Press Q to quit")

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect objects
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
