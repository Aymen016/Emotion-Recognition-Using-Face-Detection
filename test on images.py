import cv2 
import numpy as np
from keras.models import load_model

model= load_model('my_model.h5')

faceDetect = cv2.CascadeClassifier(r"C:\Users\PMLS\Downloads\archive\haarcascade_frontalface_default.xml")

labels_dict = {
    0:'Angry',
    1:'Disgust',
    2:'Fear',
    3:'Happy',
    4:'Neutral',
    5:'Sad',
    6:'Surprise'
}

# Load an image
frame = cv2.imread("img.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = faceDetect.detectMultiScale(gray, 1.3, 3)

# Process each detected face
for x, y, w, h in faces:
    # Crop the detected face
    sub_face_img = gray[y:y+h, x:x+w]
    
    # Resize the face to 48x48 pixels
    resized = cv2.resize(sub_face_img, (48, 48))
    
    # Normalize the pixel values
    normalize = resized / 255.0
    
    # Reshape the normalized face for model input
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    
    # Predict the label using the model
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    print(label)
    
    # Draw a rectangle around the detected face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # Draw additional rectangles for labeling
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
    
    # Add the label text above the rectangle
    cv2.putText(frame, labels_dict[label], (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display the image with the detections
cv2.imshow("Frame", frame)
cv2.imwrite("output_image.jpg", frame)
# Wait for a key press to close the window
cv2.waitKey(0)

# Clean up all windows
cv2.destroyAllWindows()
