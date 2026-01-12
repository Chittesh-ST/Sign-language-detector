# Real-Time Sign Language Detector

A Computer Vision and Deep Learning application that detects and translates static hand gestures into text in real-time. Built with **MediaPipe** for landmark extraction and **PyTorch** for gesture classification.

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Real%20Time-yellow)

## Project Overview
Communication gaps often exist between the deaf/hard-of-hearing community and the wider population. This project aims to bridge that gap by developing an accessible, real-time system that interprets sign language gestures using a standard webcam[1].pdf].

The system recognizes **5 distinct static gestures** and displays the corresponding label on the screen with high accuracy.

## Supported Gestures
The model is trained to recognize the following classes[1].pdf]:
1.  👋 **Hello**
2.  🙏 **Please**
3.  🤝 **Thank You**
4.  👌 **OK**
5.  👍 **Thumbs Up**

## System Architecture

The project follows a pipeline approach[1].pdf]:

1.  **Input Capture:** Webcam captures video frames in real-time.
2.  **Landmark Extraction (MediaPipe):** Detects 21 key hand landmarks (wrist, finger joints) and extracts (x, y) coordinates.
3.  **Preprocessing:** Normalizes coordinates to ensure the model is invariant to hand position within the frame.
4.  **Feature Vector:** Converts landmarks into a 42-element feature vector.
5.  **Classification (PyTorch):** A Feedforward Neural Network (FNN) predicts the gesture class.
6.  **Output:** Draws the bounding box and label on the video frame.

## Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Deep Learning:** PyTorch (Feedforward Neural Network)
* **Data Handling:** NumPy, Pickle
* **Training Interface:** Jupyter Notebooks / Scikit-Learn

## Repository Structure
```bash
├── data/                       # (Optional) Raw image dataset
├── data.pickle                 # Preprocessed landmark data
├── model.pth                   # Trained PyTorch model weights
├── model.pkl                   # (Optional) Alternative Scikit-learn model
├── 1_Model_Training.ipynb      # Notebook for data collection & training
├── 2_Real_Time_Detection.ipynb # Notebook for real-time inference
├── Project_Report.pdf          # Detailed technical report
└── requirements.txt            # Dependencies
```

## Setup & Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/sign-language-detector.git](https://github.com/your-username/sign-language-detector.git)
   cd sign-language-detector
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
1. **Run Real-Time Detection**
   
    To start the webcam and detect gestures immediately: Open 2_Real_Time_Detection.ipynb and run all cells.

    Press 'Q' to exit the webcam window.

2. **Train Your Own Model (Optional)**
   
    If you want to add new gestures or retrain the model: Open 1_Model_Training.ipynb.

     **Step 1:** Run the Data Collection cells to capture images for new classes.

     **Step 2:** Run the Preprocessing cells to generate data.pickle.

     **Step 3:** Run the Training loop to generate a new model.pth.

## Model Performance

The model utilizes a SimpleNN architecture with:
Input Layer: 42 Features (21 x,y pairs)
Hidden Layer: 64 Neurons + ReLU Activation
Output Layer: 5 Classes (Softmax)

It achieves high classification accuracy on the test set, successfully distinguishing between similar gestures by leveraging precise landmark coordinates rather than raw pixel data[1].pdf].

## License
This project is open-source and available under the MIT License.
