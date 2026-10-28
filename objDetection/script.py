import cv2
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame, verbose=False)

    # Draw bounding boxes
    annotated = results[0].plot()

    cv2.imshow("Waste Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()