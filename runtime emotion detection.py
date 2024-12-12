import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('my_model.h5')

# Start video capture
video = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary to map labels to emotions
labels_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Loop to process video frames
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
 
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

        # Display emotion label
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)

    # Display the video frame with detections
    cv2.imshow('Emotion Detector', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
