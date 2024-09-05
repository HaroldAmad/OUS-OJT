import cv2
import pickle
import numpy as np
import os
import streamlit as st
from PIL import Image

# Streamlit setup
st.header("PUP ATTENDANCE BIOMETRICS")
st.title("Face Recognition Data Collection")

# Center the rest of the elements
centered_style = """
<style>
[data-testid="stButton"], h1 , img{
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
}
</style>
"""
st.markdown(centered_style, unsafe_allow_html=True)

if 'faces_data' not in st.session_state:
    st.session_state['faces_data'] = []
    st.session_state['collecting'] = False

# Use a centered div for the rest of the elements
with st.container():

    name = st.text_input("Enter Your Name:")

    # Disable the start button if name is not provided
    start = st.button('Start Collection', disabled=(name == ""))
    stop = st.button('Stop Collection', disabled=(start==False))

    # Webcam feed placeholder
    frame_placeholder = st.empty()

    # Progress bar placeholder
    progress_bar = st.empty()

    # End of centered div
    st.markdown('</div>', unsafe_allow_html=True)

# Start the webcam feed immediately when the app opens
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('C:/Users/Harold/Documents/Github/OUS-OJT/data/haarcascade_frontalface_default.xml')

i = 0

while True:
    ret, frame = video.read()
    if not ret:
        st.write("Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

        # If collecting is active, store face data
        if st.session_state.get('collecting', False):
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(st.session_state['faces_data']) < 100 and i % 10 == 0:
                st.session_state['faces_data'].append(resized_img)

            # Update progress bar
            progress = len(st.session_state['faces_data']) / 100
            progress_bar.progress(progress)
            
            i += 1
            frame = cv2.putText(frame, str(len(st.session_state['faces_data'])), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

    # Display the live video feed in the Streamlit app
    frame_placeholder.image(frame, channels="BGR")

    if stop:
        st.session_state['collecting'] = False

    if len(st.session_state['faces_data']) >= 100:
        st.write("Collected 100 face images!")
        st.session_state['collecting'] = False
        break

    if not st.session_state.get('collecting', False):
        if start:
            st.session_state['collecting'] = True

    if not st.session_state['collecting'] and stop:
        break

video.release()
cv2.destroyAllWindows()

if st.session_state['faces_data']:
    faces_data = np.asarray(st.session_state['faces_data'])
    faces_data = faces_data.reshape(100, -1)

    data_path = 'C:/Users/Harold/Documents/Github/OUS-OJT/data'

    if 'names.pkl' not in os.listdir(data_path):
        names = [name] * 100
        with open(os.path.join(data_path, 'names.pkl'), 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(os.path.join(data_path, 'names.pkl'), 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * 100
        with open(os.path.join(data_path, 'names.pkl'), 'wb') as f:
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir(data_path):
        with open(os.path.join(data_path, 'faces_data.pkl'), 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open(os.path.join(data_path, 'faces_data.pkl'), 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open(os.path.join(data_path, 'faces_data.pkl'), 'wb') as f:
            pickle.dump(faces, f)
        
    st.write("Face data saved successfully!")
