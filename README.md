# 🔐 Sensor-Based Attendance Management & Activity Tracking System

> IoT-based smart attendance system using Fingerprint Sensor, PIR Motion Detection, Ultrasonic Distance Measurement, and Machine Learning for activity classification — with ESP32 WiFi Module and Firebase Cloud Database.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Arduino](https://img.shields.io/badge/Arduino-ESP32-green?logo=arduino)
![Firebase](https://img.shields.io/badge/Database-Firebase-orange?logo=firebase)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-yellow?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project implements an intelligent, contactless attendance management system that integrates multiple sensors with IoT and Machine Learning technologies. The system uses **fingerprint biometrics** for unique identification, **PIR and ultrasonic sensors** for presence and activity detection, and **ESP32 WiFi** for real-time cloud synchronization via Firebase.

### Key Features

- **Fingerprint Authentication** — R307/AS608 sensor with 99.9% accuracy (FAR < 0.001%)
- **Motion Detection** — PIR sensor detects physical presence before prompting scan
- **Distance Measurement** — Ultrasonic HC-SR04 measures proximity in real-time
- **Activity Classification** — ML models classify walking, sitting, standing, laying (95.45% accuracy)
- **Occupancy Detection** — Sensor-based room occupancy detection (97.71% accuracy)
- **Duplicate Prevention** — Same student cannot mark attendance twice per day
- **Real-time Cloud Sync** — Attendance data sent to Firebase Realtime Database via WiFi
- **Unauthorized Access Alerts** — Unknown fingerprints trigger security alerts
- **OLED Live Dashboard** — Real-time sensor readings displayed on 128×64 OLED

---

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Sensors   │────▶│    ESP32     │────▶│   Firebase  │────▶│  Dashboard   │
│ PIR+Ultra+  │     │  (Process +  │WiFi │  Realtime   │     │  Web/Mobile  │
│ Fingerprint │     │   WiFi)      │     │  Database   │     │  Monitoring  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

---

## 📂 Project Structure

```
attendance-ml/
├── data/
│   └── kaggle/                          # Real public datasets
│       ├── human-activity-recognition-with-smartphones/
│       │   ├── train.csv                # UCI HAR training data (7,352 samples)
│       │   └── test.csv                 # UCI HAR test data (2,947 samples)
│       ├── occupancy-detection-data-set-uci/
│       │   ├── datatraining.txt         # Occupancy training (8,143 samples)
│       │   └── datatest.txt             # Occupancy test (2,665 samples)
│       └── xAPI-Edu-Data/
│           └── xAPI-Edu-Data.csv        # Student behaviour (480 samples)
│
├── models/
│   └── real_data/
│       └── kaggle_results.json          # Trained model results & metrics
│
├── charts/                              # Generated visualization images
│   ├── chart1_accuracy_comparison.png   # Bar chart — all models compared
│   ├── chart2_best_models.png           # Horizontal bar — best per task
│   ├── chart3_confusion_matrices.png    # Confusion matrix heatmaps
│   └── chart4_heatmap.png              # All models × all tasks heatmap
│
├── wokwi_simulation/                    # ESP32 simulation files
│   ├── sketch.ino                       # Arduino code (sensors + Firebase)
│   ├── diagram.json                     # Circuit wiring (auto-layout)
│   └── libraries.txt                    # Required Arduino libraries
│
├── train_with_real_data.py              # Main ML training pipeline
├── generate_charts.py                   # Chart generation from JSON results
├── data_sources_guide.md                # Dataset sources & citations
└── README.md                            # This file
```

---

## 🔧 Hardware Components

| Component | Model | Role | Interface |
|-----------|-------|------|-----------|
| Microcontroller | ESP32 DevKit V4 | Central processing + WiFi | — |
| Fingerprint Sensor | R307 / AS608 | Biometric identification | UART (RX/TX) |
| PIR Motion Sensor | HC-SR501 | Presence detection | GPIO 27 |
| Ultrasonic Sensor | HC-SR04 | Distance measurement | GPIO 13, 12 |
| OLED Display | SSD1306 128×64 | Real-time status display | I2C (SDA=21, SCL=22) |
| LED Indicators | Green + Red | Visual feedback | GPIO 2, 4 |
| Buzzer | Passive | Audio feedback | GPIO 15 |

**Estimated Hardware Cost:** $27–52 USD

---

## 📊 ML Model Results

Trained on **real public benchmark datasets** from Kaggle / UCI ML Repository:

### Employee Activity Recognition (UCI HAR Dataset — 10,299 samples)

| Algorithm | Accuracy | F1-Score |
|-----------|----------|----------|
| Logistic Regression | **95.45%** | 0.9544 |
| Random Forest | 92.67% | 0.9266 |
| KNN (k=7) | 88.87% | 0.8877 |

### Office Occupancy Detection (UCI Occupancy Dataset — 10,808 samples)

| Algorithm | Accuracy |
|-----------|----------|
| Logistic Regression | **97.71%** |
| Random Forest | 95.35% |
| KNN | 91.97% |
| Isolation Forest | 84.20% |

### Employee Performance Classification (Employee Activity & Evaluation — 7,000 samples)

> ⚠️ **Class-imbalance note:** the `High` class contains only ~43 samples (0.6%). Model 3 uses `class_weight='balanced'` and `compute_sample_weight` to prevent the classifier from collapsing to the majority class. Macro-F1 is reported alongside accuracy.

| Algorithm | Accuracy | Macro-F1 |
|-----------|----------|----------|
| Gradient Boosting | **94.43%** | — |
| Random Forest | 92.50% | — |
| KNN (distance-weighted) | 86.21% | — |
| Decision Tree | 85.93% | — |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Arduino IDE or Wokwi Simulator
- Firebase Account (free tier)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/attendance-tracking-system.git
cd attendance-tracking-system/attendance-ml

# Install Python dependencies
py -3.12 -m pip install numpy pandas scikit-learn matplotlib seaborn
```

### Train ML Models

```bash
# Train all models + generate charts (auto-cleans old results)
py -3.12 train_with_real_data.py
```

### Generate Charts Only (from saved JSON)

```bash
py -3.12 generate_charts.py
```

### Run Simulation

1. Open [wokwi.com](https://wokwi.com)
2. Create new ESP32 project
3. Paste `wokwi_simulation/sketch.ino` into code editor
4. Paste `wokwi_simulation/diagram.json` into diagram editor
5. Add libraries from `wokwi_simulation/libraries.txt`
6. Click ▶️ Play

---

## 🔥 Firebase Integration

The system sends attendance data to Firebase Realtime Database in real-time:

```
Database URL: https://smart-attendance-and-activity-default-rtdb.asia-southeast1.firebasedatabase.app
```

### Database Structure

```json
{
  "attendance": {
    "STU-001": {
      "name": "Rahim Ahmed",
      "department": "CSE",
      "status": "present",
      "distance_cm": 42,
      "activity": "WALKING",
      "confidence": 285
    }
  },
  "alerts": {
    "latest": {
      "type": "UNAUTHORIZED_ACCESS",
      "message": "Unknown fingerprint detected!"
    }
  }
}
```

---

## 📚 Datasets

| Dataset | Source | Samples | Purpose |
|---------|--------|---------|---------|
| UCI HAR | [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) | 10,299 | Employee activity recognition |
| UCI Occupancy | [Kaggle](https://www.kaggle.com/datasets/robmarkcole/occupancy-detection-data-set-uci) | 10,808 | Office occupancy detection |
| Employee Activity & Evaluation | [Kaggle](https://www.kaggle.com/datasets/nailasayed/employee-activity-and-evaluation-dataset) | 7,000 | Employee performance classification |

---

## 📝 Citation

If you use this project in your research, please cite:

```
@thesis{attendance2026,
  title={Sensor-Based Attendance Management and Activity Tracking System using Wi-Fi Module},
  year={2026},
  institution={Bangladesh University of Professions}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for benchmark datasets
- [Wokwi](https://wokwi.com/) for ESP32 simulation platform
- [Firebase](https://firebase.google.com/) for real-time database
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms
