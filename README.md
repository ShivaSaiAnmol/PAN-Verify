# PAN-Verify
# 🪪 PAN Card Verification System

> **AI-Powered Identity Verification Using YOLO, EasyOCR & TensorFlow**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Text%20Extraction-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Project Overview

Verifying PAN cards manually can be time-consuming and error-prone. This project automates the entire verification process using Artificial Intelligence, Computer Vision, and Optical Character Recognition (OCR).

The system intelligently detects a PAN card from an uploaded image, extracts important details, validates the PAN number format, verifies the authenticity of the PAN logo, and generates a verification result within seconds.

### 🎯 Goal

Build a smart verification system that can:

✅ Detect PAN cards automatically

✅ Extract information accurately

✅ Validate PAN details

✅ Verify logo authenticity

✅ Reduce manual verification effort

---

## 🧠 How It Works

```text
User Uploads PAN Image
           │
           ▼
YOLO Detects PAN Card
           │
           ▼
EasyOCR Extracts Text
           │
           ▼
PAN Format Validation
           │
           ▼
TensorFlow Verifies Logo
           │
           ▼
Final Verification Result
```

---

## ✨ Key Features

### 🔍 PAN Card Detection

Uses YOLO Object Detection to locate and identify PAN cards from uploaded images.

### 📝 Smart Text Extraction

Extracts:

* PAN Number
* Card Holder Name
* Father's Name
* Date of Birth

using EasyOCR.

### ✅ PAN Number Validation

Checks whether the PAN follows the official format:

```text
ABCDE1234F
```

### 🏷️ Logo Authentication

A TensorFlow-based CNN model verifies whether the PAN logo is genuine.

### ⚡ Real-Time Results

Instant verification results through a web interface.

### 🌐 User-Friendly Dashboard

Simple and interactive UI built with Flask, HTML, CSS, and JavaScript.

---

## 🛠️ Tech Stack

| Category         | Technology            |
| ---------------- | --------------------- |
| Backend          | Flask                 |
| Programming      | Python                |
| Object Detection | YOLO                  |
| OCR              | EasyOCR               |
| Deep Learning    | TensorFlow            |
| Computer Vision  | OpenCV                |
| Data Processing  | NumPy, Pandas         |
| Frontend         | HTML, CSS, JavaScript |

---

## 📂 Project Architecture

```text
PAN-Verification-System
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── YOLO Model
│   ├── TensorFlow Logo Classifier
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── dataset/
```

---

## 🎯 AI Models Used

### YOLO Object Detection

Detects PAN card regions with high accuracy and speed.

### EasyOCR

Extracts text from the detected PAN card image.

### TensorFlow CNN

Classifies PAN logos as:

* Genuine
* Suspicious
* Invalid

---

## 📸 Verification Pipeline

### Step 1

📤 Upload PAN Card

↓

### Step 2

🎯 Detect Card using YOLO

↓

### Step 3

📝 Extract Text using EasyOCR

↓

### Step 4

🔍 Validate PAN Number

↓

### Step 5

🏷️ Verify Logo using CNN

↓

### Step 6

✅ Generate Final Result

---

## 📊 Expected Output

```text
PAN Number : ABCDE1234F
Name       : John Doe
DOB        : 01/01/2000

Logo Status : Genuine
PAN Format  : Valid

Verification Result :
✓ PAN Card Verified Successfully
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone <repository-url>
cd PAN-Verification-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Computer Vision
* Deep Learning
* Object Detection
* OCR Systems
* Flask Deployment
* Data Validation
* AI-Powered Document Verification


## ⭐ If you found this project useful

Give it a ⭐ on GitHub and support the project!
