import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

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

# Function to list all CSV files in the directory
def list_csv_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return files

# Directory containing the CSV files
attendance_dir = "C:/Users/Harold/Documents/Github/OUS-OJT/Attendance"

# List all CSV files
csv_files = list_csv_files(attendance_dir)

# Streamlit app
st.title("Attendance Data Viewer")

# Autorefresh every 2 seconds
from streamlit_autorefresh import st_autorefresh
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")


# Dropdown to select CSV file
selected_file = st.selectbox("Select CSV file", options=csv_files)

# Display selected CSV file's data
if selected_file:
    file_path = os.path.join(attendance_dir, selected_file)
    df = pd.read_csv(file_path)
    st.dataframe(df.style.highlight_max(axis=0))

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