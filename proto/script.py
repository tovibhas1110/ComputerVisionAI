import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Base directory
baseDir = Path(__file__).resolve().parent

# Image paths
imageS_path = baseDir / "outputStress_0.png"
imageD_path = baseDir / "outputDesign_0.png"

# Reading images
imageS = cv2.imread(str(imageS_path),cv2.IMREAD_GRAYSCALE)
imageD = cv2.imread(str(imageD_path),cv2.IMREAD_GRAYSCALE)

# Checking if the images have been loaded
if imageS is None:
    raise FileNotFoundError(f"Cannot load image: {imageS_path}")
if imageD is None:
    raise FileNotFoundError(f"Cannot load image: {imageD_path}")

# Convert to a binary image - Heaviside Function
_, binaryS = cv2.threshold(imageS, 127, 255, cv2.THRESH_BINARY)
_, binaryD = cv2.threshold(imageD, 127, 255, cv2.THRESH_BINARY)

# Operations on the Stress Image to deterime the regions that are critical
# Apply dilation
# Creating a kernel - Fiter size is 13
kernel = np.ones((13, 13), dtype=np.uint8)
dilated = cv2.dilate(binaryS,kernel,iterations=1)
# Canny filter for edge detection - Lower threshold is 50 and upper is 150
edgesS = cv2.Canny(dilated, 50, 150)

# Operations on the Design image to determine edges
# Gaussian Filter to evaluate the weighted average - fiter size 9X9
blurred = cv2.GaussianBlur(binaryD, (9, 9), 0)
# Canny filter to get the borders - Lower threshold is 50 and upper is 150
edgesD = cv2.Canny(blurred, 50, 150)


#########################################################
##Plotting the results
#########################################################
plt.figure(1)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(imageS, cmap="gray")
axes[0, 0].set_title("Stress Plot")
axes[0, 1].imshow(binaryS, cmap="gray")
axes[0, 1].set_title("Binary")
axes[0, 2].imshow(dilated, cmap="gray")
axes[0, 2].set_title("Dilated")
axes[1, 0].imshow(edgesS, cmap="gray")
axes[1, 0].set_title("Canny Edges")
axes[1, 1].imshow(edgesD, cmap="gray")
axes[1, 1].set_title("Dilated Canny Edges")

# Creating colored image to show the design and detected critical reagions 
combined = np.zeros((*edgesS.shape, 3), dtype=np.uint8)
# Red channel: original edges
combined[:, :, 0] = edgesS
# Green channel: dilated edges
combined[:, :, 1] = edgesD
# Blue channel: original design
combined[:, :, 2] = 255 - binaryD

axes[1, 2].imshow(combined)
axes[1, 2].set_title("Colored Edge Comparison")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()

############################################################
## Identifying the isolated regions critical regions
############################################################
# Get the regions that are not black
mask = (dilated > 0).astype(np.uint8)

# Identify connected regions
# numRegions - Total number of regions including background
# labels - The image is returned with regions given the labels
# stats - contain the geometrical info of the isolated regions
# centroids - centroids of the region 
numRegions, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(mask,connectivity=8))

# Convert image to color for visualization
result = cv2.cvtColor(mask * 255,cv2.COLOR_GRAY2BGR)

# Plotting the bounding box for each isolated region
for i in range(1, numRegions):

    x = stats[i, cv2.CC_STAT_LEFT]      # coordinate of bottom left corner
    y = stats[i, cv2.CC_STAT_TOP]       # coordinate of bottom left corner
    w = stats[i, cv2.CC_STAT_WIDTH]     # width of the region
    h = stats[i, cv2.CC_STAT_HEIGHT]    # height of the region    
    area = stats[i, cv2.CC_STAT_AREA]   # Area of the region in terms of pixels

    cx, cy = centroids[i]               # centroid of the region
    
    # Draw bounding rectangle
    # (image, bottomLeftCoordinate, topRightCoordinate, color, linewidth)
    cv2.rectangle(result,(x, y),(x + w, y + h),(0, 255, 0),2)

    # Write region number and area
    # (image, text, position, font, fontScale, color, thickness)
    cv2.putText(result,f"{i}: {area} px",(x, max(y - 5, 10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)

    print(
        f"Region {i}: " 
        f"Area = {area} pixels, "
        f"Width = {w}, "
        f"Height = {h}, "
        f"Centroid = ({cx:.1f}, {cy:.1f})"
    )

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Isolated Regions")
plt.axis("off")
plt.show()

############################################################
## Measuring the perimeter of the area
############################################################
# Determing the contour based on the isolated regions
# (image, external contour, simple storage to save memory)
contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Calculating the perimeters for each contour
for i, contour in enumerate(contours, start=1):
    
    perimeter = cv2.arcLength(contour, True)

    print(
        f"Region {i}: "
        f"Perimeter = {perimeter:.2f} px"
    )

