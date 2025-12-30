# Brain Tumor Detection System

Automatic brain tumor detection system based on deep learning techniques for the classification of brain MRI images using a Convolutional Neural Network (CNN).

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.95% |
| **Precision** | 89.14% |
| **Recall** | 87.03% |
| **Loss** | 0.4313 |

### Performance by Class

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 90.6% | 93.7% | 92.1% |
| Meningioma | 90.5% | 58.8% | 71.3% |
| No Tumor | 87.6% | 97.3% | 92.2% |
| Pituitary | 84.7% | 99.3% | 91.4% |

---

## Model Architecture

- **Type**: Convolutional Neural Network (CNN)
- **Structure**: 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- **Total Parameters**: 850,532
- **Input Size**: 128×128 grayscale images
- **Output**: 4 classes with Softmax activation

---

## Dataset

- **Total Images**: 7,023
  - Training: 5,712 images (80%)
  - Testing: 1,311 images (20%)
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Format**: JPG/PNG converted to grayscale (128×128)

⚠️ The dataset is not included in this repository due to size and license constraints.

---

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) or CPU
- Minimum 8 GB RAM

### Dependency Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-detection-deep-learning.git
cd brain-tumor-detection-deep-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Brain Tumor Detection System

An automatic brain tumor detection system based on Deep Learning for the classification of brain MRI images using a Convolutional Neural Network (CNN).  
This project aims to assist medical diagnosis by providing fast and reliable predictions from MRI scans.

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.95% |
| **Precision** | 89.14% |
| **Recall** | 87.03% |
| **Loss** | 0.4313 |

### Performance by Class

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 90.6% | 93.7% | 92.1% |
| Meningioma | 90.5% | 58.8% | 71.3% |
| No Tumor | 87.6% | 97.3% | 92.2% |
| Pituitary | 84.7% | 99.3% | 91.4% |

---

## Model Architecture

- **Type**: Convolutional Neural Network (CNN)
- **Structure**: 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- **Total Parameters**: 850,532
- **Input Size**: 128 × 128 grayscale MRI images
- **Output**: 4 classes with Softmax activation

---

## Dataset

- **Total Images**: 7,023
  - Training: 5,712 images (80%)
  - Testing: 1,311 images (20%)
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Format**: JPG / PNG converted to grayscale (128 × 128)

⚠️ The dataset is not included in this repository due to size and license constraints.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (recommended) or CPU
- Minimum 8 GB RAM

### Dependency Installation

```bash
# Clone the repository
git clone https://github.com/HoudaElmeski/brain-tumor-detection-deep-learning.git
cd brain-tumor-detection-deep-learning

# Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / Mac

# Install dependencies
pip install -r requirements.txt
#pour lancer app
#cd web_app
#python app.py
#http://localhost:5000
Author

Houda EL MESKI
GitHub: https://github.com/HoudaElmeski

License

This project is licensed under the MIT License.