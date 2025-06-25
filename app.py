import streamlit as st
import numpy as np
import cv2
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

uploaded_files = st.file_uploader("Upload one or more receipt images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.info("Processing images...")
    progress = st.progress(0)
    processed_data = []

    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        preprocessed = preprocess_image(img_bgr)
        progress.progress(10 + i * 15)

        cropped_img = extract_single_table_region_per_image([preprocessed])[0]
        progress.progress(25 + i * 15)

        text = extract_text_from_cropped_image(cropped_img)
        progress.progress(40 + i * 15)

        clean_text = preprocess_text(text)
        progress.progress(60 + i * 15)

        structured = fine_tune_gpt4all(clean_text)
        progress.progress(80 + i * 15)

        processed_data.append(structured)

    csv_file = save_to_csv(processed_data)
    progress.progress(100)
    st.success("✅ Processing complete!")

    with open(csv_file, "rb") as f:
        st.download_button("Download Excel/CSV", f, file_name="structured_output.csv")
