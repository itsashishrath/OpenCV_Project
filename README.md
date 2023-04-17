<h1>Overview:</h1>
<p>This repository contains three projects related to face detection and recognition. The projects are:</p>

<ul>
<li> Detection in Video</li>
<li>Face Recognition in Video</li>
<li>Face Recognition in Images</li>
<li>Face Detection in Video:</li>
</ul>
<p>This project focuses on detecting faces in a video using OpenCV and Haar Cascade classifier. The code reads a video file, captures each frame of the video, and then detects faces in each frame using the Haar Cascade classifier. The detected faces are then marked with a rectangle and displayed on the video. The project is implemented in the file face_detection_in_video.py.</p>

<h1>Face Recognition in Video:</h1>
<p>This project focuses on recognizing faces in a video using OpenCV and LBPH Face Recognizer. The project is divided into two parts, i.e., training the model and recognizing faces in the video. The training part involves reading a dataset of face images, detecting faces using Haar Cascade classifier, and training the LBPH Face Recognizer. The trained model is then saved to a file. In the recognition part, the project reads a video file, captures each frame of the video, detects faces in each frame using Haar Cascade classifier, and then recognizes the detected faces using the trained LBPH Face Recognizer. The recognized faces are then marked with a rectangle and displayed on the video. The project is implemented in the files faces_train.py and face_recognition_in_video.py.</p>

<h1>Face Recognition in Images:</h1>
<p>This project focuses on recognizing faces in images using OpenCV and LBPH Face Recognizer. The project is similar to the face recognition in video project, except that instead of reading a video file, it reads a dataset of face images and recognizes the faces in each image. The project is implemented in the files faces_train.py and face_recognition_in_images.py.</p>

Generating Trained Model:
<p>To generate the trained model, you need to run the faces_train.py file. This file reads a dataset of face images, detects faces using Haar Cascade classifier, and trains the LBPH Face Recognizer. The trained model is then saved to a file named trained.xml.

<h1>Dataset:</h1>
<a href="https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset?resource=download">Dataset Download</a>
  
<h1>Usage:</h1>
<ul>
<li>Clone the repository using the command git clone <repository_url></li>
<li>Install the required packages using the command pip install opencv-contrib-python caer</li>
<li>First edit the dataset folder location in trainer.py file </li>
<li>Run the trainer to generate model for recognising,use the command python train.py</li>
<li>The file will be saved as trained.yml</li>
<li>To run the face detection in video project, use the command python face_detection_in_video.py <video_path></li>
<li>To run the face recognition in video project, use the command python face_recognition_in_video.py <video_path></li>
<li>To run the face recognition in images project, use the command python face_recognition_in_images.py</li>
</ul>
<h1>Dependencies:</h1>
OpenCV
NumPy
Credits:
The projects are inspired by the tutorials from the OpenCV documentation and freeCodeCamp.
