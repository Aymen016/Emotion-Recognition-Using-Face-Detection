# Emotion Recognition Using Face Detection 🎭🤖

**A deep learning-based project for recognizing emotions through facial expressions in real-time! This project leverages OpenCV for face detection and a trained deep learning model for emotion classification. Perfect for exploring AI applications in human emotion analysis!**

---

## 🚀 Features
- **Real-Time Face Detection**: Detect faces from a live webcam feed using OpenCV's Haar Cascade Classifier.
- **Emotion Prediction**: Recognize emotions such as Happy, Sad, Angry, Neutral, and more using a trained convolutional neural network (CNN).
- **Efficient Preprocessing**: Grayscale conversion, normalization, and resizing for optimal model input.
- **Interactive Output**: Display detected faces and their predicted emotions directly on the webcam feed.

---

## 📂 Repository Structure
```bash
Emotion-Recognition-Using-Face-Detection/
├── model_file.h5             # Trained model file
├── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
├── train_model.ipynb         # Jupyter Notebook for training the emotion recognition model
├── real_time_emotion.py      # Main Python script for real-time emotion recognition
├── requirements.txt          # Dependencies for the project
├── README.md                 # Project documentation
└── images/                   # Sample images or screenshots of the application
