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

st.set_page_config(initial_sidebar_state="collapsed")
with st.sidebar:
    st.subheader("PUP Biometrics Attendance System")
    st.button("Home")
    st.button("Dashboard")
    st.button("Sign Out")

PUPSIDE = "pup.svg"
PUPMAIN = "pup.svg"

st.logo(
    PUPMAIN,
    icon_image=PUPSIDE,
)


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.imgur.com/E6EeiTk.jpeg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
left_co, cent_co,last_co = st.columns(3)

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
st.title("PUP Face Recognition Attendance System")

# Placeholder for the video feed
col1a, col2a, col3a, col4a, col5a, col6a, col7a= st.columns(7)
with col2a:
    stframe = st.empty()

col1, col2, col3= st.columns(3)

with col2:
    status_text = st.empty()
    # Button for taking attendance
    take_attendance = st.button('Take Attendance', key="take_attendance")


# Video capture setup
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

COL_NAMES = ['NAME', 'TIME', 'DATE']

attendance_taken = False
attendance_result = ""

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
height: 5%;
bottom: 0;
width: 100%;
background-color: maroon;
color: white;
text-align: center;
}

</style>
<div class="footer">
<p>Polytechnic University of The Philippines Biometric Attendance system | OJT</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

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
        # Crop and resize the detected face
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        label = output[0]  # Get the predicted label

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Put the label above the rectangle
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the video feed on the Streamlit app
    stframe.image(frame, channels="BGR", width=420)

    # Update status text
    if face_detected:
        status_text.write(" Valid Face detected!")
    else:
        status_text.write("Waiting for valid face")

    # Take attendance only when the button is pressed
    if take_attendance and not attendance_taken:
        if face_detected:
            (x, y, w, h) = faces[0]  # Only process the first detected face
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            label = output[0]  # Get the predicted label

            # Capture timestamp for attendance
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            attendance = [label, str(timestamp), str(date)]

            # Save attendance to CSV file
            attendance_file = "C:/Users/Harold/Documents/Github/OUS-OJT/Attendance/Attendance_" + date + ".csv"
            file_exists = os.path.isfile(attendance_file)

            with open(attendance_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

            attendance_result = "Attendance recorded for (" + label + " "+ date + timestamp + ")"
            attendance_voice = "Attendance recorded for" + label 
            speak(attendance_voice)
            st.success(attendance_result)
            attendance_taken = True  # Mark attendance as taken

video.release()
cv2.destroyAllWindows()

# Display the attendance result message after the loop
if attendance_result:
    st.write(attendance_result)
