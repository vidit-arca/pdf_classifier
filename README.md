
# PDF Classifier 

PDF Classifier is a Python-based machine learning project designed to automatically categorize PDF documents into user-defined categories (e.g., Bills, Cards, Referral Forms, Summaries, etc.). It is ideal for researchers, businesses, or organizations that regularly handle large volumes of documents and need to automate digital archiving or workflow sorting.




## Features

- Batch Classification: Automatically classifies multiple PDFs in a folder with a single command. 
- Text and OCR Extraction: Handles both digital (text-based) PDFs and scanned/image PDFs using OCR (Optical Character Recognition).
- Custom Categories: Organizes documents based on user-defined folders (categories).
- File Copying: Keeps your original documents untouched—classified copies are organized into new folders.
- GUI + Script Mode: Supports both graphical file selection (using Tkinter) and script-based folder processing.
- Performance Reporting: Outputs detailed classification metrics (accuracy, recall, f1-score).


## How It Works


Data Preparation:

- Place labeled PDF files into subfolders under a main data/ directory, with each subfolder representing a different category.

Training:

- The script extracts text or uses OCR for scanned images.
- It preprocesses the text, removes stopwords, and converts it into numeric features using a TF-IDF vectorizer.
- Trains a Logistic Regression model on the extracted features and saves the trained model for later use.

Inference/Classifying New PDFs:

- Loads the trained model and vectorizer.
- Users can select PDFs via a GUI or specify a folder for batch processing.
- The script classifies each PDF and copies it to a corresponding category folder.


## Prerequisites

- Python 3.8+ recommended

Required Python libraries:
-  pandas, scikit-learn, PyPDF2, pdf2image, pytesseract, Pillow, nltk, joblib, tqdm, tkinter

- Tesseract OCR installed (for scanned/image PDFs — see installation notes for your OS)
## Setup & Usage

Install dependencies:

- pip install -r requirements.txt
Prepare your dataset:


- data/
- ├── Bill/
- ├── Card/
- ├── Referral Form/
- └── Summary/
Each subfolder contains PDFs for that category.

Train the model:

- python main.py
- Model and vectorizer will be saved in output/.

Classify new/bulk PDFs:

Via GUI:

- python predict.py

Via Script/Folder:

- Update and run the batch classification script to classify all PDFs from a folder and copy them to category folders.
