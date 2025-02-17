# MedPro - Medicine Substitute Finder

MedPro is a web application that helps users find substitute medicines by analyzing handwritten prescriptions. The application uses OCR (Optical Character Recognition) to read medicine names from uploaded images and provides information about available substitutes, their usage, and side effects.

## Prerequisites

1. Python 3.7 or higher
2. Tesseract OCR engine installed on your system
3. The medicines dataset from Kaggle

## Installation

1. Clone this repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR:
   - For Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - For Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - For macOS: `brew install tesseract`

4. Download the medicines dataset from Kaggle and place it in the project root directory as `medicines_dataset.csv`

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Upload an image containing handwritten medicine names
4. View the substitutes and their details in the results page

## Features

- OCR-based medicine name extraction from images
- Find substitute medicines from a comprehensive database
- View usage instructions and side effects for each substitute
- Simple and intuitive web interface
- Mobile-responsive design

## Dataset

This application uses the "250k Medicines Usage, Side Effects and Substitutes" dataset from Kaggle. The dataset contains information about medicines, their substitutes, usage instructions, and side effects.

## Note

The accuracy of medicine name recognition depends on the quality of the handwriting and image. For best results:
- Ensure good lighting when taking the photo
- Write medicine names clearly and legibly
- Use high-contrast pen and paper
- Take clear, focused photos of the prescription