import streamlit as st
import tempfile
import os

st.title("MyStudyMate (MVP)")
st.write("Upload a lecture audio or PDF file")

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")
    file_extension = uploaded_file.name.split('.')[-1]

    # Save file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    st.info(f"File saved to: {temp_file_path}")