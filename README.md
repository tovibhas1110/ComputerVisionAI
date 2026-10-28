# ComputerVisionAI

**Image Processing, Computer Vision, and Geometric Feature Extraction in Topology Optimized Designs using Python and OpenCV**

## Overview

ComputerVisionAI is a Python-based project for exploring classical computer vision techniques and extracting geometric information using digital images of the stress-field indicator of topology-optimized designs.

These features play a critical role in determining the performance of topology-optimized designs. The long-term objective is to develop a methodology for optimizing build orientations in critical regions while accounting for transient thermal effects during the additive manufacturing process.

The project focuses on image preprocessing, edge detection, binary segmentation, connected-component analysis, contour extraction, and quantitative shape measurements.

## Features

The project explores the following techniques:

- **Image preprocessing:** Grayscale conversion and Gaussian filtering.
- **Binary segmentation:** Threshold-based separation of foreground and background.
- **Edge detection:** Canny edge detection and image gradients.
- **Morphological operations:** Dilation and other operations for refining binary regions.
- **Connected-component analysis:** Identification and labeling of isolated regions.
- **Contour detection:** Extraction of object boundaries.
- **Geometrical measurements:** Area, perimeter, bounding boxes, and centroids.
- **Visualization:** Annotated images and comparison plots using Matplotlib.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV (`opencv-python`) | Image processing and geometric analysis |
| NumPy | Numerical and array operations |
| Matplotlib | Image visualization and plotting |
| Ultralytics YOLO | Exploration of deep-learning-based object detection |
| Git and GitHub | Version control and project management |

## Image Processing Workflow

The classical image-processing workflow is:

```text
Input Image
     |
     v
Grayscale Conversion
     |
     v
Gaussian Blur
     |
     v
Thresholding / Canny Edge Detection
     |
     v
Morphological Operations
     |
     v
Connected Components / Contour Detection
     |
     v
Geometrical Feature Extraction
     |
     v
Visualization and Analysis
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tovibhas1110/ComputerVisionAI.git
cd ComputerVisionAI
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install opencv-python numpy matplotlib
```

For experiments involving YOLO object detection:

```bash
python -m pip install ultralytics
```

## Project Status

**In development — learning and prototyping.**

The repository documents experiments with classical computer vision and the development of an image-analysis workflow.

## References

- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Ultralytics Documentation](https://docs.ultralytics.com/)

## Author

**Vibhas Mishra**

[GitHub Profile](https://github.com/tovibhas1110)
