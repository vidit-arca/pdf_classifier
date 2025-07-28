import os
import shutil
from main import extract_text_from_pdf, preprocess_text
import joblib

clf = joblib.load("output/clf.pkl")
vectorizer = joblib.load("output/vectorizer.pkl")

def classify_pdf(pdf_path):
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        return None
    clean_text = preprocess_text(raw_text)
    X = vectorizer.transform([clean_text])
    prediction = clf.predict(X)
    return prediction[0]

def classify_and_copy_pdfs_in_folder(input_folder, output_base_folder):
    for name in os.listdir(input_folder):
        full_path = os.path.join(input_folder, name)

        # Skip if it's a directory; only process files
        if not os.path.isfile(full_path):
            print(f"Skipped directory or non-file item: {name}")
            continue

        if name.lower().endswith('.pdf'):
            pred = classify_pdf(full_path)
            if pred:
                target_folder = os.path.join(output_base_folder, pred)
            else:
                target_folder = os.path.join(output_base_folder, "Unclassified")
            os.makedirs(target_folder, exist_ok=True)
            target_path = os.path.join(target_folder, name)
            shutil.copy2(full_path, target_path)
            print(f"Copied '{name}' to '{target_folder}'")

if __name__ == '__main__':
    input_folder = "/Users/apple/Desktop/pdf_classifier/data1/35478055"
    output_base_folder = "/Users/apple/Desktop/pdf_classifier/classified_output"
    classify_and_copy_pdfs_in_folder(input_folder, output_base_folder)
