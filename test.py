from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Function to generate speech
def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Load face detection model and KNN model data
facedetect = cv2.CascadeClassifier('C:/Users/Harold/Documents/Github/OUS-OJT/data/haarcascade_frontalface_default.xml')

with open('C:/Users/Harold/Documents/Github/OUS-OJT/data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('C:/Users/Harold/Documents/Github/OUS-OJT/data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize KNN classifier and fit the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Streamlit interface components
st.title("Face Recognition Attendance System")

# Button for taking attendance
take_attendance = st.button('Take Attendance', key="take_attendance")

# Placeholder for the video feed
stframe = st.empty()
status_text = st.empty()

# Video capture setup
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

COL_NAMES = ['NAME', 'TIME', 'DATE']

attendance_taken = False
attendance_result = ""

# Continuously display the video feed
while True:
    ret, frame = video.read()
    if not ret:
        st.error("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around detected faces
    face_detected = False
    for (x, y, w, h) in faces:
        face_detected = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Display the video feed on the Streamlit app
    stframe.image(frame, channels="BGR")

    # Update status text
    if face_detected:
        status_text.write("Face detected!")
    else:
        status_text.write("No face detected!")

    # Take attendance only when the button is pressed
    if take_attendance and not attendance_taken:
        if face_detected:
            (x, y, w, h) = faces[0]  # Only process the first detected face
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)

            # Capture timestamp for attendance
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            attendance = [str(output[0]), str(timestamp), str(date)]

            # Save attendance to CSV file
            attendance_file = "C:/Users/Harold/Documents/Github/OUS-OJT/Attendance/Attendance_" + date + ".csv"
            file_exists = os.path.isfile(attendance_file)

            with open(attendance_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

            attendance_result = "Attendance recorded for " + str(output[0])
            speak(attendance_result)
            st.success(attendance_result)
            attendance_taken = True  # Mark attendance as taken

video.release()
cv2.destroyAllWindows()

# Display the attendance result message after the loop
if attendance_result:
    st.write(attendance_result)
