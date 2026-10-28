from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Input image
image_path = BASE_DIR / "images" / "outputStress_0.png"

# Output directory
output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)

# Load image
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"Cannot load image: {image_path}"
    )

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Perform object detection
results = model.predict(
    source=image,
    conf=0.25,
    verbose=False
)

# Get detection results
result = results[0]

# Draw bounding boxes
annotated_image = result.plot()

# Print detected objects
print("\nDetected objects:")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    object_name = result.names[class_id]

    print(
        f"Object: {object_name}, "
        f"Confidence: {confidence:.2f}"
    )

# Save output image
output_path = output_dir / "detected_waste.jpg"

cv2.imwrite(
    str(output_path),
    annotated_image
)

# Display original and detected images
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)
axes[0].set_title("Original Image")

axes[1].imshow(
    cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
)
axes[1].set_title("Detected Objects")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print(f"\nResult saved to: {output_path}")
