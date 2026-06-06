# Potato Chip Defect Detector 🔍

Real-time potato chip quality inspection using YOLOv8 computer vision.

## Setup

### 1. Python version
Python 3.12.7 is required.

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment
Windows:
```bash
venv\Scripts\activate
```
Mac/Linux:
```bash
source  venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Place model file
Put your `best.pt` file in the same folder as `app.py`.

```
chip_detector_demo/
├── app.py
├── best.pt          ← your trained model here
├── requirements.txt
└── README.md
```

### 6. Run the app
```bash
streamlit run app.py
```

The dashboard opens automatically at http://localhost:8501

## Usage

- Select **Image** mode to run detection on a single photo
- Select **Video** mode to process a full video file
- Adjust the **confidence threshold** in the sidebar (lower = more detections)
- Download annotated results using the export button

## Classes
- 🟢 `whole_chip` — intact chip
- 🔴 `broken_chip` — defective/broken chip 

## Model
- Architecture: YOLOv8n (nano)
- Training: Fine-tuned on custom potato chip dataset
- mAP50: ~0.97
