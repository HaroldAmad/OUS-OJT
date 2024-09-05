import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

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
