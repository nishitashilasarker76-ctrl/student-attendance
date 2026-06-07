# 📊 Data Sources Guide — Where to Get Data for Your Thesis

## আপনার প্রজেক্টে ৩ ধরনের Data দরকার:

| # | Task | কি Data দরকার | কোথা থেকে পাবেন |
|---|------|--------------|-----------------|
| 1 | 🏃 Activity Classification | Accelerometer x,y,z + Activity labels | UCI HAR Dataset (Kaggle) |
| 2 | 🚪 Occupancy/Presence Detection | PIR, Temperature, Light, CO2 | UCI Room Occupancy Dataset |
| 3 | 📅 Attendance Records | Student ID, date, time, status | Kaggle Attendance + Your synthetic data |

---

## ✅ Dataset 1: UCI HAR (Human Activity Recognition)
- **Link:** https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
- **Size:** 10,299 samples, 561 features
- **Activities:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- **Sensors:** Accelerometer (x,y,z) + Gyroscope (x,y,z) from smartphone
- **Subjects:** 30 volunteers, ages 19-48
- **Use for:** Activity classification (your Model 1)
- **License:** Public domain
- **Citation:** Anguita et al., 2013

## ✅ Dataset 2: Room Occupancy Estimation (UCI)
- **Link:** https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
- **Size:** 10,000+ data points
- **Features:** Temperature, Light, Sound, CO2, PIR motion, Room_Occupancy_Count
- **Use for:** Presence detection, anomaly detection (your Model 2)
- **License:** CC BY 4.0

## ✅ Dataset 3: PIRvision Presence Detection (UCI)
- **Link:** https://archive.ics.uci.edu/dataset/1101/pirvision_fog_presence_detection
- **Size:** 55 PIR sensor readings per sample
- **Classes:** Vacancy (0), Stationary presence (1), Motion (3)
- **Use for:** PIR-based presence detection

## ✅ Dataset 4: School Attendance (Kaggle)
- **Link:** https://www.kaggle.com/datasets/thajegan76/attendance
- **Size:** Multiple Excel files, daily attendance records
- **Use for:** Attendance prediction model

## ✅ Dataset 5: Student Performance & Behaviour (Kaggle)
- **Link:** https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- **Size:** 480 students, 16 features including attendance
- **Use for:** Attendance prediction + behaviour analysis

## ✅ Dataset 6: Your Synthetic Data (Already generated!)
- **File:** data/sensor_readings.csv (3,780 records)
- **File:** data/attendance_records.csv (215 records)
- **Use for:** System-specific testing + supplementary data

---

## 🎯 Recommended Strategy for Thesis:
1. **Primary:** UCI HAR dataset → Activity classification (REAL data!)
2. **Primary:** UCI Room Occupancy → Presence/anomaly detection (REAL data!)
3. **Supporting:** Kaggle Attendance → Attendance prediction
4. **Supplementary:** Your synthetic data → System-specific testing
5. **Paper:** "We used publicly available benchmark datasets from UCI ML Repository and Kaggle..."

## 📝 How to cite in paper:
```
[1] UCI HAR Dataset: D. Anguita et al., "A Public Domain Dataset for Human Activity 
    Recognition Using Smartphones," ESANN 2013.
[2] Room Occupancy: A.P. Singh et al., "Room Occupancy Estimation," 
    UCI ML Repository, 2023.
```
