# Real-Time Motorcycle Helmet Detection

Computer vision system utilizing the YOLOv8 architecture to monitor motorcycle helmet compliance. The project encompasses the full development pipeline, from manual dataset annotation to a functional web-based deployment.

## Technical Performance & Analysis

### Metrics
* **mAP@0.5:** 0.771 (77.1%)
* **Compliance Accuracy:** 86% detection rate for riders wearing helmets.
* **Non-Compliance Risk:** 18% of violators were misidentified as compliant.
* **False Positives:** 54% background misidentification rate, primarily due to geometric similarities between helmets and environmental objects (e.g., mirrors, bags).

### Key Observations
* **High Recall:** The model successfully prioritizes "safety-first" detection, minimizing missed detections of actual helmets.
* **Training Stability:** Classification loss curves indicate stable differentiation between human heads and safety gear.
* **Optimal Threshold:** A confidence threshold of 0.48 is recommended to balance precision and recall during live inference.

### Development Roadmap
* **Negative Sampling:** Integrate non-helmet round objects (basketballs, mirrors) into the training set to reduce geometric over-generalization.
* **Class Balancing:** Expand the "No Helmet" dataset with diverse hair textures and headwear to close the 21% accuracy gap between classes.
* **Contextual Verification:** Implement hierarchical detection to verify human presence before executing the helmet detection head.

## Project Structure
* **Data Preparation:** Dataset labeling and augmentation ($640 \times 640$ resizing, grayscale, and brightness filters) via Roboflow.
* **Training:** YOLOv8 (Nano) architecture trained over 50 epochs.
* **Evaluation:** Performance validation using Confusion Matrices and Precision-Recall curves.
* **Deployment:** Live webcam inference delivered through a Streamlit dashboard.

