import streamlit as st
import numpy as np
import os
from PIL import Image

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

if uploaded_files:
    st.info("Processing images...")
    progress = st.progress(0)
    processed_data = []

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
        csv_path = save_to_csv([structured], filename=filename_base + ".csv") 

        progress.progress(min(95 + i * 2, 100))

        with open(csv_path, "rb") as f:
            st.download_button(
                label=f"Download CSV for {uploaded_file.name}",
                data=f,
                file_name=os.path.basename(csv_path),
                mime="text/csv"
            )

    st.success("✅ All files processed!")
