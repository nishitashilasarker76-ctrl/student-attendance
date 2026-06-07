"""
=============================================================
STEP 1: Generate Synthetic Sensor Data
=============================================================
যেহেতু আমাদের real sensor নেই, আমরা realistic synthetic data 
তৈরি করব যা real sensor থেকে আসা data-র মতো দেখতে হবে।

Sensors simulated:
  - PIR (motion: 0/1)
  - Ultrasonic (distance in cm)
  - RFID (card UID)
  - Accelerometer (x, y, z values)
  - Entry/Exit IR beam
=============================================================
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# ---- Configuration ----
NUM_DAYS = 30           # 30 দিনের data
NUM_USERS = 10          # 10 জন student/employee
OUTPUT_DIR = "data"

# ---- Registered Users ----
users = [
    {"uid": "A1B2C3D4", "name": "Rahim Ahmed",    "id": "STU-001", "dept": "CSE"},
    {"uid": "E5F6A7B8", "name": "Fatima Khan",     "id": "STU-002", "dept": "CSE"},
    {"uid": "C9D0E1F2", "name": "Karim Hossain",   "id": "STU-003", "dept": "EEE"},
    {"uid": "1A2B3C4D", "name": "Nasrin Akter",    "id": "STU-004", "dept": "CSE"},
    {"uid": "5E6F7A8B", "name": "Shakib Rahman",   "id": "STU-005", "dept": "EEE"},
    {"uid": "9C0D1E2F", "name": "Aisha Begum",     "id": "STU-006", "dept": "BBA"},
    {"uid": "3A4B5C6D", "name": "Imran Hasan",     "id": "STU-007", "dept": "CSE"},
    {"uid": "7E8F9A0B", "name": "Sumaiya Islam",   "id": "STU-008", "dept": "EEE"},
    {"uid": "B1C2D3E4", "name": "Tanvir Alam",     "id": "STU-009", "dept": "BBA"},
    {"uid": "F5A6B7C8", "name": "Maliha Rahman",   "id": "STU-010", "dept": "CSE"},
]

# ---- Unknown UIDs (for anomaly detection) ----
unknown_uids = ["XX1122YY", "ZZ3344AA", "UNKNOWN01", "HACKER99"]

def generate_attendance_data():
    """Generate daily attendance records"""
    records = []
    start_date = datetime(2026, 1, 1, 8, 0, 0)
    
    for day in range(NUM_DAYS):
        current_date = start_date + timedelta(days=day)
        
        # Skip weekends (Friday-Saturday for Bangladesh)
        if current_date.weekday() in [4, 5]:  # Friday=4, Saturday=5
            continue
        
        for user in users:
            # 85% chance of attending
            if random.random() < 0.85:
                # Entry time: 8:00 - 9:30 AM (normal) or 9:30 - 11:00 (late)
                is_late = random.random() < 0.2  # 20% chance of being late
                if is_late:
                    entry_hour = random.randint(9, 10)
                    entry_min = random.randint(0, 59)
                else:
                    entry_hour = random.randint(8, 9)
                    entry_min = random.randint(0, 30)
                
                entry_time = current_date.replace(hour=entry_hour, minute=entry_min, 
                                                   second=random.randint(0, 59))
                
                # Exit time: 3:00 - 5:30 PM (normal) or 1:00 - 2:30 PM (early leave)
                is_early_leave = random.random() < 0.1  # 10% early leave
                if is_early_leave:
                    exit_hour = random.randint(13, 14)
                    exit_min = random.randint(0, 59)
                else:
                    exit_hour = random.randint(15, 17)
                    exit_min = random.randint(0, 30)
                
                exit_time = current_date.replace(hour=exit_hour, minute=exit_min,
                                                  second=random.randint(0, 59))
                
                duration_hours = (exit_time - entry_time).seconds / 3600
                
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "user_uid": user["uid"],
                    "user_name": user["name"],
                    "user_id": user["id"],
                    "department": user["dept"],
                    "entry_time": entry_time.strftime("%H:%M:%S"),
                    "exit_time": exit_time.strftime("%H:%M:%S"),
                    "duration_hours": round(duration_hours, 2),
                    "is_late": 1 if is_late else 0,
                    "is_early_leave": 1 if is_early_leave else 0,
                    "status": "present",
                    "entry_method": random.choice(["RFID", "RFID", "RFID", "Fingerprint"]),
                })
            else:
                # Absent
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "user_uid": user["uid"],
                    "user_name": user["name"],
                    "user_id": user["id"],
                    "department": user["dept"],
                    "entry_time": "",
                    "exit_time": "",
                    "duration_hours": 0,
                    "is_late": 0,
                    "is_early_leave": 0,
                    "status": "absent",
                    "entry_method": "",
                })
        
        # Add some unauthorized access attempts (anomalies)
        if random.random() < 0.3:  # 30% chance per day
            anomaly_hour = random.choice([6, 7, 21, 22, 23, 1, 2])  # Unusual hours
            anomaly_time = current_date.replace(hour=anomaly_hour, 
                                                 minute=random.randint(0, 59))
            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "user_uid": random.choice(unknown_uids),
                "user_name": "UNKNOWN",
                "user_id": "UNAUTHORIZED",
                "department": "UNKNOWN",
                "entry_time": anomaly_time.strftime("%H:%M:%S"),
                "exit_time": "",
                "duration_hours": 0,
                "is_late": 0,
                "is_early_leave": 0,
                "status": "unauthorized",
                "entry_method": "RFID",
            })
    
    return records


def generate_sensor_data():
    """Generate raw sensor readings (PIR, Ultrasonic, Accelerometer)"""
    records = []
    start_date = datetime(2026, 1, 1, 8, 0, 0)
    
    for day in range(NUM_DAYS):
        current_date = start_date + timedelta(days=day)
        if current_date.weekday() in [4, 5]:
            continue
        
        # Generate readings every 5 minutes from 7:00 AM to 10:00 PM
        for hour in range(7, 22):
            for minute in range(0, 60, 5):
                timestamp = current_date.replace(hour=hour, minute=minute)
                
                # Office hours: 8 AM - 5 PM → more activity
                is_office_hours = 8 <= hour <= 17
                
                # PIR motion probability
                if is_office_hours:
                    pir_motion = 1 if random.random() < 0.7 else 0
                else:
                    pir_motion = 1 if random.random() < 0.05 else 0  # Very rare outside hours
                
                # Ultrasonic distance (cm) — closer = person nearby
                if pir_motion:
                    distance = random.uniform(10, 100)  # Person detected 10-100cm
                else:
                    distance = random.uniform(200, 400)  # No one nearby
                
                # Accelerometer (simulating door/gate movement)
                if pir_motion and random.random() < 0.3:
                    accel_x = random.uniform(-2.0, 2.0)
                    accel_y = random.uniform(-2.0, 2.0)
                    accel_z = random.uniform(8.0, 11.0)  # Gravity + movement
                    activity = random.choice(["walking", "walking", "standing", "sitting", "running"])
                else:
                    accel_x = random.uniform(-0.1, 0.1)
                    accel_y = random.uniform(-0.1, 0.1)
                    accel_z = random.uniform(9.7, 9.9)  # Just gravity
                    activity = "idle"
                
                # IR beam (entry/exit count)
                ir_entry = 1 if (pir_motion and random.random() < 0.15) else 0
                ir_exit = 1 if (pir_motion and random.random() < 0.1) else 0
                
                # Zone (which area the sensor is in)
                zone = random.choice(["entrance", "classroom_A", "classroom_B", "lab", "library"])
                
                # Anomaly flag
                is_anomaly = 0
                if not is_office_hours and pir_motion:
                    is_anomaly = 1  # Motion outside office hours = anomaly
                if distance < 15 and not pir_motion:
                    is_anomaly = 1  # Very close but no PIR = sensor error
                
                records.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "minute": minute,
                    "zone": zone,
                    "pir_motion": pir_motion,
                    "ultrasonic_cm": round(distance, 1),
                    "accel_x": round(accel_x, 3),
                    "accel_y": round(accel_y, 3),
                    "accel_z": round(accel_z, 3),
                    "ir_entry": ir_entry,
                    "ir_exit": ir_exit,
                    "activity_label": activity,
                    "is_office_hours": 1 if is_office_hours else 0,
                    "is_anomaly": is_anomaly,
                })
    
    return records


def save_csv(records, filename):
    """Save records to CSV file"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not records:
        print(f"No records to save for {filename}")
        return
    
    keys = records[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"✅ Saved {len(records)} records → {filepath}")


def generate_user_registry():
    """Save registered users"""
    filepath = os.path.join(OUTPUT_DIR, "registered_users.csv")
    keys = ["uid", "name", "id", "dept"]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(users)
    print(f"✅ Saved {len(users)} users → {filepath}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("📊 STEP 1: Generating Synthetic Sensor Data")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Attendance data
    print("\n📋 Generating attendance records...")
    attendance_data = generate_attendance_data()
    save_csv(attendance_data, "attendance_records.csv")
    
    # 2. Raw sensor data
    print("\n📡 Generating raw sensor data...")
    sensor_data = generate_sensor_data()
    save_csv(sensor_data, "sensor_readings.csv")
    
    # 3. User registry
    print("\n👥 Generating user registry...")
    generate_user_registry()
    
    # Summary
    present = sum(1 for r in attendance_data if r["status"] == "present")
    absent = sum(1 for r in attendance_data if r["status"] == "absent")
    unauth = sum(1 for r in attendance_data if r["status"] == "unauthorized")
    anomalies = sum(1 for r in sensor_data if r["is_anomaly"] == 1)
    
    print("\n" + "=" * 60)
    print("📊 DATA SUMMARY:")
    print(f"   Attendance records : {len(attendance_data)}")
    print(f"     ├── Present      : {present}")
    print(f"     ├── Absent       : {absent}")
    print(f"     └── Unauthorized : {unauth}")
    print(f"   Sensor readings    : {len(sensor_data)}")
    print(f"     └── Anomalies    : {anomalies}")
    print(f"   Registered users   : {len(users)}")
    print("=" * 60)
    print("✅ All data saved to ./data/ folder!")
