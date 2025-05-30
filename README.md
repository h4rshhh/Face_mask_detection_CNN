# Realtime Face Mask Detection using custom CNN and OpenCV
![stuuupegg-mask](https://github.com/user-attachments/assets/01e7445b-a8ff-4dce-a6be-370f51e6f339)

A deep learning system for Real time face-mask detection 
## 🎯 About Project

This project uses a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get an accuracy of 96.2% on the training set and 94.3% on the test set. 
Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using _OpenCV_. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction.

The model is capable of predicting multiple faces with or without masks at the same time

## ✨ Key Features

- **High Accuracy**: 96.2% training accuracy and 94.3% test accuracy
- **Real-time Performance**: 30 FPS inference on CPU-only systems
- **Fast Face Detection**: Under 20ms per frame using _Haar Cascade Classifier_
- **No GPU Dependency**: Optimized for CPU performance
- **Live Webcam Integration**: Seamless OpenCV integration for real-time detection

## 🔧 Technical Details

### Model Architecture
- Custom Convolutional Neural Network (CNN)
- Trained on 7,500+ labeled images
- Optimized for real-time inference

### Performance Metrics
- **Training Accuracy**: 96.2%
- **Test Accuracy**: 94.3%
- **Inference Speed**: 30 FPS
- **Face Detection Time**: <20ms per frame

### Technologies Used
- **Deep Learning**: Custom CNN architecture
- **Computer Vision**: OpenCV for real-time processing
- **Face Detection**: Haar Cascade Classifier
- **Programming Language**: Python




## 📊 Dataset

The model was trained on a comprehensive dataset of 7,500+ labeled images containing faces with and without masks.

**Dataset Source**: [[face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)]

## 🔬 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 96.2% |
| Test Accuracy | 94.3% |
| Inference Speed | 30 FPS |
| Face Detection Time | <20ms |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Dependencies
- OpenCV (`cv2`)
- TensorFlow/Keras
- NumPy
- Matplotlib (for visualization)

### Usage

1. **Clone the repository**:
```bash
git clone https://github.com/h4rshhh/Realtime-Face-Mask-Detection.git
cd Realtime-Face-Mask-Detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run real-time detection**:
```bash
python detect.py
```
