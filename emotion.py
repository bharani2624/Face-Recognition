# import cv2
# from keras.models import load_model
# import numpy as np

# # Load the pre-trained Keras model from .h5 file
# model = load_model("face_model.h5")

# # Haar cascade file for face detection
# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)

# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0

# # Open the webcam (index 0)
# webcam = cv2.VideoCapture(0)

# # Dictionary to map predicted labels to emotions
# labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# while True:
#     ret, im = webcam.read()  # Read from webcam
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
#     faces = face_cascade.detectMultiScale(im, 1.3, 5)  # Detect faces in the grayscale image
    
#     try:
#         for (p, q, r, s) in faces:
#             face_image = gray[q:q+s, p:p+r]  # Extract face region from grayscale image
#             cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  # Draw rectangle around the face
#             face_image_resized = cv2.resize(face_image, (48, 48))  # Resize face image to model input size
#             img = extract_features(face_image_resized)  # Extract features and normalize
#             pred = model.predict(img)  # Make prediction
#             prediction_label = labels[pred.argmax()]  # Get predicted label
#             cv2.putText(im, '%s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))  # Put label on the image
#         cv2.imshow("Output", im)  # Display the output image
#         if cv2.waitKey(27) & 0xFF == ord('q'):  # Exit when 'q' is pressed
#             break
#     except cv2.error as e:
#         print(f"OpenCV error: {e}")
#         break

# webcam.release()  # Release the webcam
# cv2.destroyAllWindows()  # Close all OpenCV windows
from facial_emotion_recognition import EmotionRecognition
import cv2

# Load the pre-trained model
emotion_recognizer = EmotionRecognition(device='gpu')

# Open webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    frame = emotion_recognizer.recognise_emotion(frame)  # Detect emotion
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
