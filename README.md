# Image Forgery Detection and Classification Web Application

A Flask-based web application that detects whether an image is authentic or forged and, if forged, classifies the forgery type.

## Features

- Binary classification: Authentic vs Forged
- Forgery type classification:
  - Copy-Move Forgery
  - Splicing Forgery
- Confidence score for predictions
- Inference time measurement
- Simple web interface using Flask

---

## Project Structure

```text
forgery_site/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── (Place trained model files here)
│
└── templates/
    └── index.html
```

---

## Model Files

The trained model files are not included in this repository because of GitHub file size limitations.

Place the following files inside the `models/` directory before running the application:

```text
models/
├── best_forgery_v5.keras
└── best_forgery_detector_v2.keras
```

You can download the trained models from:

https://drive.google.com/drive/folders/1P82NeNlMJCl-Bl7gEJ19S9P4B2XCJFgI?usp=sharing

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShafiyaMasrath/Minor_Project_Image_Forgery.git
cd Minor_Project_Image_Forgery
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model files

Download the trained `.keras` files and place them inside the `models/` folder.

### 4. Run the application

```bash
python app.py
```

---

## Usage

1. Open your browser.
2. Navigate to:

```text
http://localhost:5000
```

3. Upload an image.
4. Click **Analyse Image**.
5. View:
   - Authenticity prediction
   - Confidence score
   - Forgery type (if forged)
   - Inference time

---

## Supported Image Formats

- JPG
- JPEG
- PNG
- BMP
- TIFF

---

## Technology Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Pillow

---

## Performance

### Binary Forgery Detection Model
- Accuracy: 91.15%

### Forgery Type Classification Model
- Classes:
  - Copy-Move
  - Splicing

---

## Future Improvements

- Forgery localization using segmentation masks
- Attention-based architectures
- Explainable AI visualizations
- Multi-dataset evaluation
- Cloud deployment

---

## Author

**Shafiya Masrath**
