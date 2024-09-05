import cv2
import pickle
import numpy as np
import os
import streamlit as st
from streamlit_navigation_bar import st_navbar
from streamlit_autorefresh import st_autorefresh

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
st.subheader("PUP Biometrics Attendance System")
st.markdown(page_bg_img, unsafe_allow_html=True)


    

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

# Start the webcam feed
col1a, col2a, col3a, col4a, col5a, col6a, col7a= st.columns(7)
with col2a:
    frame_placeholder = st.empty()
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    facedetect = cv2.CascadeClassifier('C:/Users/Harold/Documents/Github/OUS-OJT/data/haarcascade_frontalface_default.xml')


# Use a centered div for the rest of the elements
with st.container():
    name = st.text_input("Enter Your Name:")
    col1, col2, col3 , col4 = st.columns(4)
    with col2 :
        start = st.button('Start Collection', disabled=(name == ""))
    with col3 :
        stop = st.button('Stop Collection', disabled=(start==False))
    
    progress_bar = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Refresh the page if Stop Collection is pressed
    if stop:
        st.session_state['collecting'] = False
        st.session_state['faces_data'] = []
        st.rerun()

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
    frame_placeholder.image(frame, channels="BGR", width=480)

    

    # Check button states
    if start and not st.session_state['collecting']:
        st.session_state['collecting'] = True
    if len(st.session_state['faces_data']) >= 100:
        st.write("Collected 100 face images!")
        st.session_state['collecting'] = False
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


    
