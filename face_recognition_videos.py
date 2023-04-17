import cv2
import os

people = []
dataset_path = 'celebrities'

for folder_name in os.listdir(dataset_path):
    people.append(str(folder_name))

# Load the trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained.yml')

# Load the face detection classifier
face_detector = cv2.CascadeClassifier('haar_face.xml')

# Open the local video file
cap = cv2.VideoCapture('output.mp4')

# Loop through each frame in the video
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Use the recognizer to predict the label for the face ROI
        label, confidence = recognizer.predict(face_roi)
        
        # Draw a rectangle around the detected face
        color = (0, 255, 0) # Green
        thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Add the predicted label and confidence as text to the frame
        text = f'{people[label]} ({confidence:.2f})'
        org = (x, y-10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        cv2.putText(frame, text, org, font, font_scale, color, thickness)
    
    # Display the current frame
    cv2.imshow('Face Recognition', frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
