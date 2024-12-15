# 🎭 Emotion Recognition Using Face Detection 🤖

**A deep learning-based project for recognizing emotions through facial expressions in real-time! This project leverages OpenCV for face detection and a trained deep learning model for emotion classification. Perfect for exploring AI applications in human emotion analysis!**

---

## 🚀 Features
- **Real-Time Face Detection**: Detect faces from a live webcam feed using OpenCV's Haar Cascade Classifier.📹
- **Emotion Prediction**: Recognize emotions such as Happy, Sad, Angry, Neutral, and more using a trained convolutional neural network (CNN).😄 😢 😡 😐
- **Efficient Preprocessing**: Grayscale conversion, normalization, and resizing for optimal model input.🔄
- **Interactive Output**: Display detected faces and their predicted emotions directly on the webcam feed.💡
- **Flask Web App**: A web interface to interact with the emotion recognition model, providing a user-friendly way to test real-time emotion detection.🌐

---

## 📂 Repository Structure
```bash
Emotion-Recognition-Using-Face-Detection/
├── data/                             # Dataset folder
├── static/                           # Static files (CSS, JS, etc.)
├── templates/                        # HTML templates for Flask app
├── uploads/                          # Uploaded files for processing
├── runtime_emotion_detection.py      # Script for real-time detection
├── model.py                          # Training and model definition
├── my_model.h5                       # Pre-trained model
├── haarcascade_frontalface_default.xml  # Haar Cascade file
├── app.py                            # Main Flask app
├── README.md                         # Project documentation
└── requirements.txt                  # Dependencies for the project
```

---

## 🛠️ Tools and Technologies
- **Programming Language**: Python 🐍  
- **Deep Learning Framework**: TensorFlow/Keras  
- **Computer Vision Library**: OpenCV  
- **Numerical Computing**: NumPy  
- **Visualization**: Matplotlib (optional)  
- **Face Detection**: Haar Cascade Classifier  

---

## 💻 Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Emotion-Recognition-Using-Face-Detection.git
   cd Emotion-Recognition-Using-Face-Detection

2. **Install Dependencies**:  
   Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
3. **Download Haar Cascade File**:
   Ensure the `haarcascade_frontalface_default.xml` file is in the project directory. If not, download it from the [OpenCV GitHub repository]  (https://github.com/opencv/opencv/tree/master/data/haarcascades).

4. **Run the Application**:
  Start the real-time emotion recognition script:
  ```bash
    python real_time_emotion.py
  ```
---

### 📊 Model Overview  
The emotion recognition model is a Convolutional Neural Network (CNN) trained on a dataset of facial expressions. It processes grayscale images resized to 48x48 pixels for efficient and accurate emotion classification.

**Predicted Emotions:**
- Happy 😊
- Sad 😢
- Angry 😡
- Neutral 😐
- Surprise 😲

#### Model Accuracy:
The model achieves an accuracy of **75%** on the FER-2013 test dataset. While this provides a solid foundation for recognizing emotions from facial expressions, there's room for improvement. We aim to fine-tune the model for better real-world performance.

---
### 🎯 Future Enhancements
- ✨ **Add More Emotions**: Train the model to recognize additional emotions like Fear, Disgust, etc.
- 🔧 **Improve Accuracy**: Fine-tune the model for better real-world performance.
- 👥 **Multi-Face Detection**: Extend the application to predict emotions for multiple faces simultaneously.
- 🌐 **Web Integration**: Create a web-based interface for wider accessibility.

---

### 🧠 Dataset Used
The model was trained using the **FER-2013** dataset, which contains labeled facial expressions.  
[Learn more about the dataset here](https://www.kaggle.com/datasets/msambare/fer2013).

---

### 📸 Demo
Real-Time Emotion Detection in Action  
> The image taken:
![Screenshot 2024-11-27 222705](https://github.com/user-attachments/assets/28d6ede8-8382-44ad-8977-95869adc2337)

> After emotion detection:
![Screenshot 2024-11-27 222626](https://github.com/user-attachments/assets/88a0c741-426d-465b-85aa-ac04c0d77888)

---

### 📸 Demo - Flask Web App
Real-Time Emotion Detection through the Flask interface

The web app interface:
![Screenshot 2024-12-02 012110](https://github.com/user-attachments/assets/25e4fbee-ccea-4d5b-8d3a-a679b95a22b6)
![WhatsApp Image 2024-12-03 at 11 38 05 PM](https://github.com/user-attachments/assets/f057e542-0744-4fa9-84c0-7cd6f4f076ef)
![Screenshot 2024-12-02 022122](https://github.com/user-attachments/assets/cfacb0ef-25fb-4f2a-9027-01c67a745ca3)
![Screenshot 2024-12-02 022150](https://github.com/user-attachments/assets/e8ec4e64-d680-4634-bb3c-e07840996185)


### 🤝 Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements, bug fixes, or new features.

