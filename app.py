import streamlit as st
import numpy as np
import os
from PIL import Image
import streamlit.components.v1 as components

from model import (
    preprocess_image,
    extract_single_table_region_per_image,
    extract_text_from_cropped_image,
    preprocess_text,
    fine_tune_gpt4all,
    save_to_csv
)

st.title("AI Document Parser")

uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
    st.session_state.download_ready = []

    st.info("Processing images...")
    progress = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        preprocessed = preprocess_image(img_np)
        progress.progress(10 + i * 15)

        cropped_img = extract_single_table_region_per_image(preprocessed)
        progress.progress(30 + i * 15)

        text = extract_text_from_cropped_image(cropped_img)
        progress.progress(50 + i * 15)

        clean_text = preprocess_text(text)
        progress.progress(70 + i * 15)

        structured = fine_tune_gpt4all(clean_text)
        progress.progress(85 + i * 15)

        filename_base = os.path.splitext(uploaded_file.name)[0]
        csv_path = save_to_csv(structured) 

        progress.progress(100)

        st.session_state.processed_data.append((filename_base, csv_path))
        st.session_state.download_ready.append(True)

    st.success("✅ All files processed!")


def resetpage():
    components.html("""
        <script>
            window.location.reload();
        </script>
    """, height=0)


if 'download_ready' in st.session_state:
    for i, (filename, path) in enumerate(st.session_state.processed_data):
        with open(path, "rb") as f:
            if st.download_button(f"Download CSV for {filename}", f, file_name=f"{filename}.csv", key=f"dl_{i}"):
                st.session_state.clear()
                resetpage()
