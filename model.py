import os
import csv
import cv2
import numpy as np
import easyocr
import re
import tempfile
from gpt4all import GPT4All

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

def fine_tune_gpt4all(extracted_data):
    prompt = f'''Construct csv  with the following data: {extracted_data}. The data is extracted text from a invoice table. Return the table as a strict csv. Strickly Do nott give any comments, just return the csv. below is an example of how the value is supposed to returned
        coloumn1, coloumn2...<newline symbol>
        value1_1, value2_1...<newline symbol>
        value1_2, value2_2...<newline symbol>
    '''
    with model.chat_session():
        response = model.generate(prompt, max_tokens=1024)
    for line in response.strip().splitlines():
        if ',' in line and not line.lower().startswith("here"):
            return line.strip()
    

def save_to_csv(structured_data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='', encoding='utf-8')
    writer = csv.writer(temp_file)

    for line in structured_data.strip().split('\n'):
        row = [cell.strip() for cell in line.split(',')]
        writer.writerow(row)

    temp_file.close()
    return temp_file.name

def preprocess_image(img):
    h, w = img.shape[:2]
    scale = max(400 / h, 400 / w)
    if scale > 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    coords = np.column_stack(np.where(contrast > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if 1.5 <= abs(angle) <= 15:
        (h, w) = contrast.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        contrast = cv2.warpAffine(contrast, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cropped = contrast[y:y + h, x:x + w]
    else:
        cropped = contrast

    return cropped

reader = easyocr.Reader(['en'], gpu=False)

def extract_single_table_region_per_image(img, min_words=3):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped_rows = []

    for c in sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1]):
        x, y, w, h = cv2.boundingRect(c)
        if h > 15 and w > 50:
            cropped = img[y:y+h, x:x+w]
            cropped_rows.append((cropped, w * h))

    best_crop = None
    max_words = 0

    for crop, _ in cropped_rows:
        result = reader.readtext(crop, detail=0, paragraph=False)
        if len(result) > max_words:
            max_words = len(result)
            best_crop = crop

    if best_crop is not None and max_words >= min_words:
        return best_crop
    elif cropped_rows:
        fallback_crop = max(cropped_rows, key=lambda x: x[1])[0]
        return fallback_crop
    else:
        return img

def extract_text_from_cropped_image(cropped_img):
    reader = easyocr.Reader(['en'])
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    cv2.imwrite(temp_file.name, cropped_img)

    results = reader.readtext(temp_file.name)
    text = " ".join([result[1] for result in results])

    temp_file.close()
    os.remove(temp_file.name)
    return text

def preprocess_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9.,: ]', '', text)
    text = text.lower()

    return text