# Image Forgery Detector — Web App

## Folder Structure

```
forgery_site/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── best_forgery_v5.keras           ← your Phase 2 model (91.15%)
│   └── best_forgery_detector_v2.keras  ← your type classification model
└── templates/
    └── index.html
```

## Setup (Run Once)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy your models from Google Drive
Download these two files from your Google Drive and place them in the `models/` folder:
- `best_forgery_v5.keras`
- `best_forgery_detector_v2.keras`

### 3. Run the server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## How it works

1. Upload any image (JPG, PNG, BMP, TIFF)
2. Click "Analyse Image"
3. Results show:
   - **Verdict**: Authentic or Forged
   - **Confidence**: How sure the model is (%)
   - **Forgery Type**: Copy-Move or Splicing (only shown if forged)
   - **Inference Time**: How long it took in milliseconds

## Notes
- First prediction is slow (~5–10s) because TF warms up the model
- Subsequent predictions are much faster (~1–3s)
- Works best on JPEG images (same compression format as training data)
