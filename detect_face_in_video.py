import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haar_face.xml')

# Initialize the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw a bounding box around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Exit the program if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
