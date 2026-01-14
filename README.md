# Autonomous Navigation Simulation

A modular autonomous driving system that implements a perception-planning-control pipeline for lane detection and steering estimation using deep learning.

## 🚀 Overview

This project simulates an autonomous vehicle's navigation system by processing video footage to predict steering directions. It uses a hybrid branching Convolutional Neural Network (CNN) that captures both spatial features from individual frames and temporal features from sequences of frames.

## 🛠 Features

- **Perception-Planning-Control Pipeline**: A standard modular architecture for autonomous systems.
- **Deep Learning Steering Estimation**: Predicts `left`, `straight`, or `right` based on video sequences.
- **Branching CNN Architecture**:
  - **Spatial Branch**: Extracts features from the latest frame using 2D convolutions.
  - **Temporal Branch**: Captures motion dynamics over a 5-frame sequence using 3D convolutions.
- **Modular Codebase**: Organized into `perception`, `planning`, `control`, `learning`, and `utils`.

## 📁 Project Structure

```text
├── dataset/                # Video files (.MOV) for training and testing
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/                # Helper scripts
├── src/
│   ├── control/            # Steering and velocity control logic
│   ├── learning/           # Model definition and dataset processing
│   ├── perception/         # Lane detection and computer vision tasks
│   ├── planning/           # Path planning algorithms
│   ├── utils/              # Video processing and general utilities
│   └── visualization/      # Rendering and debug visualization
└── run.py                  # Main entry point for training and evaluation
```

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/shaheerfarid/autonomous-navigation-simulation.git
   cd autonomous-navigation-simulation
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
   _(Note: Ensure `keras`, `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`, and `opencv-python` are installed.)_

## 🚀 Usage

### Training the Model

To train the model on the provided dataset and evaluate its performance:

```bash
python run.py
```

The script will:

1. Load and format the video dataset.
2. Train the branching CNN model or load an existing `.keras` model.
3. Save the trained model as `autonomous-navigation-simulation.keras`.
4. Run predictions on test instances and display the results.

## 🧠 Model Architecture

The core of the system is a **Branching CNN Model** designed for efficiency (< 10M parameters):

- **Input**: A sequence of 5 grayscale images (240x320x1).
- **Spatial Branch (2D CNN)**: Processes only the most recent frame to identify current lane markers and obstacles.
- **Temporal Branch (3D CNN)**: Processes the full 5-frame sequence to understand the vehicle's trajectory and motion.
- **Merge & Classification**: The features are concatenated and passed through dense layers to output probabilities for 3 classes: `left`, `straight`, and `right`.

## 📊 Dataset

The dataset consists of video files located in the `dataset/` directory. The `format_dataset` utility processes these videos, packing consecutive frames into training samples and mapping steering values (dx) into classification labels.

---

Developed as part of the Autonomous Navigation Simulation project.
