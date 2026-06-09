# 📊 Data Sources Guide — Where to Get Data for Your Thesis

> A comprehensive, thesis-oriented reference for sourcing, understanding, preprocessing, and citing the datasets that power this Sensor-Based Attendance Management & Activity Tracking System. Read this before downloading anything — it will save you hours of searching.

---

## 🧭 Overview — আপনার প্রজেক্টে কী কী Data দরকার

Your thesis has **three ML modeling tasks** plus one end-to-end IoT data flow. Each task benefits from a different public dataset. Below is the full mapping:

| # | Task | কি Data দরকার | Primary Dataset | Backup / Alternative |
|---|------|--------------|-----------------|----------------------|
| 1 | 🏃 Activity Classification | Accelerometer x,y,z + Gyro x,y,z + Activity label | **UCI HAR** (Kaggle) | WISDM, PAMAP2, MotionSense |
| 2 | 🚪 Occupancy / Presence Detection | PIR, Temperature, Light, CO2, Sound | **UCI Room Occupancy** | PIRvision Fog, ASHRAE energy |
| 3 | 🎓 Student Behaviour / Attendance Prediction | Demographics + academic features + attendance outcome | **xAPI-Edu-Data** (Kaggle) | "Student Alcohol Consumption", Predict students' dropout |
| 4 | 📅 Attendance Log Format (synthetic) | Student ID, date, time, status | **Your generated** `data/attendance_records.csv` | Kaggle "School Attendance" |
| 5 | 📡 Sensor Stream (synthetic, for IoT pipeline) | PIR, Ultrasonic, Fingerprint ID, activity | **Your generated** `data/sensor_readings.csv` | Wokwi live capture |

---

## 📑 Quick-Jump Table of Contents

1. [Dataset 1 — UCI HAR (Human Activity Recognition)](#-dataset-1--uci-har-human-activity-recognition)
2. [Dataset 2 — UCI Room Occupancy Estimation](#-dataset-2--uci-room-occupancy-estimation)
3. [Dataset 3 — PIRvision Fog Presence Detection](#-dataset-3--pirvision-fog-presence-detection)
4. [Dataset 4 — School Attendance (Kaggle)](#-dataset-4--school-attendance-kaggle)
5. [Dataset 5 — xAPI-Edu-Data (Student Performance)](#-dataset-5--xapi-edu-data-student-performance--behaviour)
6. [Dataset 6 — Your Synthetic Sensor & Attendance Data](#-dataset-6--your-synthetic-sensor--attendance-data)
7. [Alternative / Backup Datasets by Task](#-alternative--backup-datasets-by-task)
8. [Recommended Strategy for Thesis](#-recommended-strategy-for-thesis)
9. [Preprocessing & Loading Recipes](#-preprocessing--loading-recipes)
10. [Licensing, Ethics & Privacy Notes](#-licensing-ethics--privacy-notes)
11. [How to Cite in Paper (BibTeX + IEEE)](#-how-to-cite-in-paper)
12. [Data Source Checklist (printable)](#-data-source-checklist-printable)

---

## ✅ Dataset 1 — UCI HAR (Human Activity Recognition)

The flagship dataset for your **Model 1 (Activity Classification)** — produces 95.45% accuracy with Logistic Regression on your pipeline.

- **Kaggle link:** https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
- **UCI link:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Size:** 10,299 samples (7,352 train / 2,947 test, **pre-split by subject** — never mix train/test subjects!)
- **Features:** 561 engineered time/frequency features per sample (already extracted from raw signals)
- **Raw signals available:** Yes — `Inertial Signals/` folder contains 9-channel raw triaxial accelerometer + gyroscope
- **Window:** 2.56 s sliding window, 50% overlap, 128 readings/window, 50 Hz sampling
- **Activities (6 classes):**
  1. `WALKING`
  2. `WALKING_UPSTAIRS`
  3. `WALKING_DOWNSTAIRS`
  4. `SITTING`
  5. `STANDING`
  6. `LAYING`
- **Sensors:** Smartphone (Samsung Galaxy S II) worn on the waist
- **Subjects:** 30 volunteers, ages 19–48
- **Class balance:** Reasonably balanced (~1,372–1,906 samples per class)
- **Use for:** Activity classification — your **Model 1** / `train_with_real_data.py` task 1
- **License:** Public Domain (CC0-equivalent)
- **Citation:** D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz, *"A Public Domain Dataset for Human Activity Recognition Using Smartphones"*, ESANN 2013.

### Column-level schema (train.csv)

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `subject` | int | 1 | Subject ID 1–30 — useful for leave-one-subject-out CV |
| `Activity` | string | `WALKING` | Target label |
| `tBodyAcc-mean()-X` | float | 0.288 | Body accel mean on X |
| `tBodyAcc-mean()-Y` | float | -0.0193 | Body accel mean on Y |
| `tBodyAcc-mean()-Z` | float | -0.117 | Body accel mean on Z |
| … 558 more `tBodyAcc*/tGravityAcc*/tBodyGyro*/fBodyAcc*/fBodyGyro*/angle/*` features | float | various | Time + frequency domain |
| `angle(X,gravityMean)` | float | 0.013 | Useful engineered angle features |

> 💡 **Tip for thesis:** If you only have the Kaggle pre-split `train.csv`/`test.csv`, you still get a clean 70/30 split. Do **not** re-split — that would leak subject information across folds and inflate accuracy.

---

## ✅ Dataset 2 — UCI Room Occupancy Estimation

The dataset behind your **Model 2 (Presence / Occupancy Detection)** — 97.71% accuracy with Logistic Regression.

- **UCI link:** https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
- **Kaggle mirror:** https://www.kaggle.com/datasets/robmarkcole/occupancy-detection-data-set-uci
- **Size:** 10,808 samples across 3 rooms; `datatraining.txt` (8,143), `datatest.txt` (2,665), `datatest2.txt` (9,752)
- **Sensors:** Temperature, Humidity, Light, CO2, Humidity Ratio
- **Target:** `Room_Occupancy_Count` (integer 0–3) or `Occupancy` (binary 0/1)
- **Sampling rate:** ~1 minute
- **Rooms covered:** 1 office room, 1 small conference room, 1 larger lecture room
- **Use for:** Presence / anomaly detection — your **Model 2** in `train_with_real_data.py`
- **License:** CC BY 4.0

### Column-level schema

| Column | Type | Range | Role |
|--------|------|-------|------|
| `date` | datetime (YYYY-MM-DD HH:MM:SS) | 2015-02-04 → 2015-02-18 | Time index |
| `Temperature` | float °C | 19.0 – 26.2 | Feature |
| `Humidity` | float % | 16.7 – 63.9 | Feature |
| `Light` | float lux | 0 – 1544.5 | Feature — strongest single predictor |
| `CO2` | float ppm | 412 – 2076 | Feature — most informative for human presence |
| `HumidityRatio` | float | 0.0027 – 0.0078 | Derived feature |
| `Occupancy` (Room_Occupancy_Count) | int 0–3 | 0 = empty | **Target** |

> ⚠️ **Be careful:** `datatest.txt` is from **Room 1 only**, while `datatest2.txt` covers a different room. The Kaggle `occupancy-detection-data-set-uci` repo concatenates both. When you report accuracy in your thesis, state which split you used.

---

## ✅ Dataset 3 — PIRvision Fog Presence Detection

A more focused dataset for **PIR-only occupancy detection** — useful if your thesis wants to defend the choice of using **only PIR + ultrasonic** (no CO2 / light) as a low-cost, privacy-preserving alternative.

- **UCI link:** https://archive.ics.uci.edu/dataset/1101/pirvision_fog_presence_detection
- **Size:** 1,800+ samples; 55 PIR sensor readings per sample (5 PIR sensors × 11 features)
- **Classes:** `Vacancy (0)`, `Stationary presence (1)`, `Motion (3)`
- **Use for:** Validating that a *cheap* PIR-only setup can distinguish "person in the room still" from "person in the room moving" from "no one"
- **License:** CC BY 4.0

> 🎯 **When to cite this in your paper:** Include it as a "supporting benchmark" demonstrating that PIR-array signals (similar to your HC-SR501 + HC-SR04 combo) are sufficient for occupancy detection — without invading privacy via cameras or CO2 sensing.

---

## ✅ Dataset 4 — School Attendance (Kaggle)

Real-world attendance log format — use it to demonstrate that your Firebase database schema matches what real schools use.

- **Kaggle link:** https://www.kaggle.com/datasets/thajegan76/attendance
- **Format:** Multiple Excel files, daily attendance records
- **Use for:** Schema validation, comparison of your `attendance_records.csv` format against real institutional data
- **License:** Varies by file — check Kaggle dataset page before publishing thesis

> ⚠️ This dataset is **not used for training** in your pipeline. It is a reference for table structure and date/time format.

---

## ✅ Dataset 5 — xAPI-Edu-Data (Student Performance & Behaviour)

The dataset for your **Model 3 (Student Behaviour Classification)** — SVM achieves 64.58% on it.

- **Kaggle link:** https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- **UCI link:** https://archive.ics.uci.edu/dataset/342/student+performance
- **Size:** 480 students, 16 features
- **Use for:** Attendance prediction + behaviour analysis (low/medium/high performing students)
- **License:** Public Domain

### Column-level schema

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `gender` | categorical | `M` / `F` | |
| `NationalITy` | categorical | `KW` / `Jordan` | Country |
| `PlaceofBirth` | categorical | `Jordan` | |
| `StageID` | categorical | `LowerLevel`, `MiddleSchool`, `HighSchool` | Grade level |
| `GradeID` | categorical | `G-04`, `G-07`, `G-10` | |
| `SectionID` | categorical | `A`, `B`, `C` | |
| `Topic` | categorical | `Math`, `English`, `Biology` | Subject |
| `Semester` | categorical | `F` / `S` | Fall/Spring |
| `Relation` | categorical | `Father` / `Mum` | Parent in household |
| `raisedhands` | int 0–100 | 25 | Times student raised hand in class |
| `VisITedResources` | int 0–100 | 30 | Times student visited course resources |
| `AnnouncementsView` | int 0–100 | 17 | Times student checked announcements |
| `Discussion` | int 0–100 | 20 | Times student joined discussions |
| `ParentAnsweringSurvey` | categorical | `Yes` / `No` | |
| `ParentschoolSatisfaction` | categorical | `Good` / `Bad` | |
| `StudentAbsenceDays` | categorical | `Above-7` / `Under-7` | **Closest proxy to attendance** |
| `Class` | categorical | `L`, `M`, `H` | **Target** — Low/Mid/High performance |

> 📝 **Thesis tip:** `StudentAbsenceDays` is the column that maps cleanly to your "attendance prediction" framing. If the supervisor pushes for higher accuracy, also try binary classification (`H` vs not-`H`) and multi-class with class-weighting — class imbalance is the main reason SVM tops out at 64.58%.

---

## ✅ Dataset 6 — Your Synthetic Sensor & Attendance Data

Already generated and stored in `data/`. Use these for **end-to-end pipeline testing** of the IoT → Firebase → ML flow.

| File | Records | Schema Highlights | Used By |
|------|---------|-------------------|---------|
| `data/sensor_readings.csv` | 3,780 | `timestamp`, `pir_state`, `distance_cm`, `fingerprint_id`, `activity`, `confidence` | `train_with_real_data.py` for the "synthetic" branch and the `wokwi_simulation` smoke tests |
| `data/attendance_records.csv` | 215 | `student_id`, `name`, `date`, `time`, `status`, `distance_cm`, `activity` | Validates the Firebase push schema |

> 🛠 **How it was generated:** See the comment header at the top of `train_with_real_data.py` (the `SYNTHETIC_*` constants). Distribution: ~60% present, ~25% late, ~15% absent; activity mix mirrors UCI HAR; PIR/distance values reflect HC-SR501 / HC-SR04 noise envelopes.

---

## 🔁 Alternative / Backup Datasets by Task

If a primary dataset becomes unavailable, or you want to strengthen the thesis with a *second* benchmark, use these:

### For Activity Recognition
| Dataset | Samples | Sensors | Why it's a good backup |
|---------|---------|---------|------------------------|
| **WISDM** | 1,098,207 | Smartphone accelerometer, 20 Hz | 36 users, 6 activities — bigger than UCI HAR |
| **PAMAP2** | 2,844,868 | IMU over 3 body locations + heart rate | 9 subjects, 12 activities — covers daily-life + sport |
| **MotionSense** | 4,800,000 | iPhone 6s, accelerometer + gyroscope | 24 subjects, 6 activities — same labels as UCI HAR |
| **Opportunity** | ~700,000 | 15 body-worn sensors | 4 subjects, 17 gestures — high-fidelity but complex |

### For Occupancy / Presence Detection
| Dataset | Samples | Sensors | Why it's a good backup |
|---------|---------|---------|------------------------|
| **ASHRAE Great Energy Predictor** | 19M+ | CO2, temp, weather | Building-scale, not just one room |
| **Wifilogs occupancy** | 10,000 | Wi-Fi probe requests | Privacy-preserving alternative to cameras |
| **CO2-PIR combined study** | 6,000 | CO2 + PIR | Directly mirrors your ESP32 stack |

### For Student Behaviour / Attendance
| Dataset | Samples | Features | Why it's a good backup |
|---------|---------|----------|------------------------|
| **UCI Student Performance** | 649 | 33 | Portuguese secondary school — math + Portuguese language |
| **Predict students' dropout and academic success** | 4,424 | 36 | Higher-ed dropout classification — binary |
| **Open University Learning Analytics (OULAD)** | 32,593 | 18 | Massive, time-series engagement + outcome |

---

## 🎯 Recommended Strategy for Thesis

A defensible, supervisor-friendly plan that balances **real data**, **synthetic data**, and **novelty**:

1. **Primary ML evidence (real, public, citable):**
   - UCI HAR → activity classification (Model 1, 95.45%)
   - UCI Room Occupancy → presence detection (Model 2, 97.71%)
   - xAPI-Edu-Data → behaviour classification (Model 3, 64.58%)
2. **End-to-end IoT validation (synthetic but realistic):**
   - Your `sensor_readings.csv` → ESP32 → Firebase → ML pipeline
3. **Optional second benchmark (strengthens thesis):**
   - PIRvision or WISDM as a "supporting benchmark" — show that the model generalises
4. **Paper wording:**
   > "We evaluated the system on three public benchmark datasets from the UCI Machine Learning Repository and Kaggle: UCI HAR (n=10,299), UCI Room Occupancy (n=10,808), and xAPI-Edu-Data (n=480), supplemented by 3,780 synthetically generated sensor readings for end-to-end IoT pipeline validation."

---

## 🛠 Preprocessing & Loading Recipes

Drop-in code snippets that match the loader in `train_with_real_data.py`.

### 1. Load UCI HAR (Kaggle version)
```python
import pandas as pd

train = pd.read_csv("data/kaggle/human-activity-recognition-with-smartphones/train.csv")
test  = pd.read_csv("data/kaggle/human-activity-recognition-with-smartphones/test.csv")

X_train, y_train = train.drop(columns=["Activity", "subject"]), train["Activity"]
X_test,  y_test  = test.drop(columns=["Activity", "subject"]),  test["Activity"]
# UCI HAR is already StandardScaler-normalised — do not re-scale.
```

### 2. Load UCI Room Occupancy
```python
cols = ["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]
train = pd.read_csv("data/kaggle/occupancy-detection-data-set-uci/datatraining.txt", names=cols, header=0)
test  = pd.read_csv("data/kaggle/occupancy-detection-data-set-uci/datatest.txt",      names=cols, header=0)

X_train, y_train = train[["Temperature","Humidity","Light","CO2","HumidityRatio"]], train["Occupancy"]
X_test,  y_test  = test[["Temperature","Humidity","Light","CO2","HumidityRatio"]],  test["Occupancy"]
```

### 3. Load xAPI-Edu-Data
```python
df = pd.read_csv("data/kaggle/xAPI-Edu-Data/xAPI-Edu-Data.csv")
y = df["Class"].map({"L":0, "M":1, "H":2})          # Low/Mid/High
X = pd.get_dummies(df.drop(columns=["Class"]), drop_first=True)
```

### 4. Load your synthetic data
```python
sensor_df      = pd.read_csv("data/sensor_readings.csv",      parse_dates=["timestamp"])
attendance_df  = pd.read_csv("data/attendance_records.csv",   parse_dates=["date"])
```

---

## 🛡 Licensing, Ethics & Privacy Notes

| Dataset | License | Personal Data? | Commercial Use OK? | Action Required |
|---------|---------|----------------|--------------------|-----------------|
| UCI HAR | Public Domain | No (anonymous volunteers) | ✅ Yes | Cite the ESANN paper |
| UCI Room Occupancy | CC BY 4.0 | No | ✅ Yes with attribution | Cite Singh et al. 2023 |
| PIRvision | CC BY 4.0 | No | ✅ Yes with attribution | Cite UCSD authors |
| School Attendance (thajegan76) | Per-file — check | Possibly (school names) | ⚠ Check per file | Read license on each .xlsx |
| xAPI-Edu-Data | Public Domain | Aggregated, no PII | ✅ Yes | Cite Amrieh et al. 2015 |
| Your synthetic | Yours | No (fake) | ✅ Yes | Not required, but recommended |

> 🚨 **For thesis submission:** Add a short paragraph in your Methodology chapter titled *"Data Sources and Ethics"* stating that no human subjects were recruited, all datasets are public, and the synthetic dataset was generated procedurally (no real biometric data was collected).

---

## 📚 How to Cite in Paper

### BibTeX
```bibtex
@inproceedings{anguita2013har,
  title  = {A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author = {Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge Luis},
  booktitle = {21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year   = {2013},
  pages  = {437--442}
}

@article{singh2023occupancy,
  title  = {Room Occupancy Estimation},
  author = {Singh, Aditya Pal and Jain, Vivek and Chaudhari, Sachin and Frank, Frank and K{\"o}hler, Nico and others},
  journal= {UCI Machine Learning Repository},
  year   = {2023}
}

@article{amrieh2015educational,
  title  = {Educational Data Mining & Students' Performance Prediction},
  author = {Amrieh, E. A. and Hamtini, T. and Aljarah, I.},
  journal= {International Journal of Advanced Computer Science and Applications},
  volume = {6},
  number = {9},
  year   = {2015}
}
```

### IEEE-style numeric citations (for printed thesis)
```
[1] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz,
    "A public domain dataset for human activity recognition using
    smartphones," in Proc. 21st Eur. Symp. Artif. Neural Netw.
    (ESANN), 2013, pp. 437-442.

[2] A. P. Singh, V. Jain, S. Chaudhari, F. Frank, and N. Köhler,
    "Room Occupancy Estimation," UCI Machine Learning Repository,
    2023. [Online]. Available: https://archive.ics.uci.edu/dataset/864

[3] E. A. Amrieh, T. Hamtini, and I. Aljarah, "Educational data
    mining and students' performance prediction," Int. J. Adv.
    Comput. Sci. Appl., vol. 6, no. 9, pp. 1-6, 2015.
```

---

## ✅ Data Source Checklist (printable)

Use this when you start writing Chapter 3 (Methodology) of your thesis:

- [ ] UCI HAR downloaded from Kaggle into `data/kaggle/human-activity-recognition-with-smartphones/`
- [ ] UCI Room Occupancy downloaded from UCI into `data/kaggle/occupancy-detection-data-set-uci/`
- [ ] xAPI-Edu-Data downloaded from Kaggle into `data/kaggle/xAPI-Edu-Data/`
- [ ] (Optional) PIRvision downloaded for supporting benchmark
- [ ] (Optional) WISDM or MotionSense downloaded as second activity-recognition benchmark
- [ ] `data/sensor_readings.csv` present (3,780 records)
- [ ] `data/attendance_records.csv` present (215 records)
- [ ] `train_with_real_data.py` runs end-to-end without errors
- [ ] `models/real_data/kaggle_results.json` regenerated with all three tasks
- [ ] `charts/chart1_accuracy_comparison.png` … `chart4_heatmap.png` regenerated
- [ ] BibTeX entries for all primary datasets added to `references.bib`
- [ ] "Data Sources and Ethics" paragraph written in Methodology chapter
- [ ] License table (above) included as Table 3.x in the thesis
- [ ] Each dataset URL tested on the day of submission (Kaggle/UCI links sometimes rotate)

---

## 📞 When a dataset is unavailable

If any link above returns 404 on the day of your defense:

1. **UCI datasets** are mirrored on multiple Kaggle datasets — search `"<dataset name> site:kaggle.com"`.
2. **OpenML** (https://www.openml.org) is a reliable secondary source for UCI datasets; search by ID:
   - UCI HAR → OpenML ID 1478
   - UCI Occupancy → OpenML ID 1494
3. **Zenodo** (https://zenodo.org) often hosts authors' personal copies of the same data.
4. **Web of Science / Google Scholar** the original paper — authors usually provide a download link in the paper itself.

---

*Last reviewed: 2026-06-09 — keep this guide in sync with `train_with_real_data.py` whenever a new dataset is added.*
