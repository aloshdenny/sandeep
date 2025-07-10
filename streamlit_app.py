import streamlit as st
import requests
import os

infer_url = os.getenv("INFER_URL")

st.title("Uniform Detection App")

# Prompt input
prompt = st.text_input(
    "Prompt",
    value="Is there a person in the image, and are they wearing a uniform? Respond with YES or NO"
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and prompt:
    if st.button("Check"):
        # Prepare files and data for the POST request
        files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        data = {"prompt": prompt}

        with st.spinner("Analyzing image..."):
            response = requests.post(infer_url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Response: {result}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")