import os
import cv2 as cv
import numpy as np

people = []
dataset_path = 'celebrities'

for folder_name in os.listdir(dataset_path):
    people.append(str(folder_name))


haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_images = []
labels = []

def create_training_data():
    for person in people:
        person_dir = os.path.join(data_dir, person)
        label = people.index(person)

        for image_filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_filename)

            image = cv.imread(image_path)
            if image is None:
                continue 

            grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                face_image = grayscale_image[y:y+h, x:x+w]
                face_images.append(face_image)
                labels.append(label)

create_training_data()
print('Training data created ---------------')

face_images = np.array(face_images, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the face_images and labels
face_recognizer.train(face_images, labels)

face_recognizer.save('face_trained.yml')

np.save('face_images.npy', face_images)
np.save('labels.npy', labels)
