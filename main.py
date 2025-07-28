import os
import re
import string
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Directories
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Try to extract text directly. If empty, fallback to OCR extraction."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"

        if not text.strip():  # No text extracted, try OCR
            print(f"No text found in {pdf_path} via direct extraction. Trying OCR...")
            text = extract_text_ocr(pdf_path)
        return text

    except Exception as e:
        print(f"Failed to extract text from {pdf_path} due to {e}. Trying OCR...")
        return extract_text_ocr(pdf_path)

def extract_text_ocr(pdf_path):
    """Convert PDF pages to images and run OCR on each image."""
    text = ""
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Could not convert PDF to images for OCR: {e}")
        return ""

    for i, img in enumerate(images):
        try:
            txt = pytesseract.image_to_string(img)
            text += txt + "\n"
        except Exception as e:
            print(f"OCR failed on page {i+1} of {pdf_path}: {e}")
    return text

def preprocess_text(text):
    # Normalize whitespace, lowercase
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

def load_data(data_dir):
    texts = []
    labels = []
    categories = [cat for cat in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cat))]
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        for filename in tqdm(os.listdir(category_dir), desc=f"Loading '{category}'"):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(category_dir, filename)
                text = extract_text_from_pdf(file_path)
                processed_text = preprocess_text(text)
                if processed_text.strip():  # Only add if some text was extracted
                    texts.append(processed_text)
                    labels.append(category)
                else:
                    print(f"Warning: No text extracted after preprocessing for file {file_path}")
    return texts, labels

def main():
    print("Loading and preprocessing data...")
    texts, labels = load_data(DATA_DIR)

    print(f"Loaded {len(texts)} documents.")

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels)

    print("Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    import joblib
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "clf.pkl"))
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "vectorizer.pkl"))
    print(f"Model and vectorizer saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
