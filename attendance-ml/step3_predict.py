"""
=============================================================
STEP 3: Make Predictions with Trained Models
=============================================================
Train করা model গুলো দিয়ে এখন prediction করব!

  - নতুন sensor data দিলে activity classify করবে
  - Anomaly detect করবে
  - কোন student আগামীকাল আসবে predict করবে
=============================================================
"""

import pickle
import json
import math
from collections import Counter


# ============================================
# Need class definitions for pickle to load
# ============================================
class SimpleKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _distance(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def predict_one(self, x):
        distances = [(self._distance(x, x_train), y) 
                     for x_train, y in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda d: d[0])
        k_nearest = [d[1] for d in distances[:self.k]]
        counter = Counter(k_nearest)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for p, actual in zip(predictions, y) if p == actual)
        return correct / len(y)


class SimpleAnomalyDetector:
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.means = []
        self.stds = []
    
    def fit(self, X_normal):
        n_features = len(X_normal[0])
        self.means = []
        self.stds = []
        for i in range(n_features):
            values = [x[i] for x in X_normal]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            self.means.append(mean)
            self.stds.append(max(std, 0.001))
    
    def predict_one(self, x):
        z_scores = [abs((x[i] - self.means[i]) / self.stds[i]) 
                    for i in range(len(x))]
        avg_z = sum(z_scores) / len(z_scores)
        return 1 if avg_z > self.threshold else 0
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]


class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def _predict_one(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict_one(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict_one(node['right'], row)
            else:
                return node['right']
    
    def predict(self, X):
        return [self._predict_one(self.tree, x) for x in X]


# ============================================
# Prediction functions
# ============================================
def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def predict_activity(model, accel_x, accel_y, accel_z, pir_motion, distance):
    features = [accel_x, accel_y, accel_z, pir_motion, distance]
    prediction = model.predict_one(features)
    return prediction


def detect_anomaly(model, hour, pir, distance, ir_entry, ir_exit, is_office):
    features = [hour, pir, distance, ir_entry, ir_exit, is_office]
    result = model.predict_one(features)
    return "🚨 ANOMALY!" if result == 1 else "✅ Normal"


def predict_attendance(model, day_of_week, dept, att_rate, late_rate, day, month):
    features = [day_of_week, dept, att_rate, late_rate, day, month]
    prediction = model.predict([features])[0]
    return "✅ Will Attend" if prediction == 1 else "❌ Likely Absent"


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔮 STEP 3: Making Predictions!")
    print("=" * 60)
    
    # Load models
    print("\n📂 Loading trained models...")
    knn = load_model("models/knn_activity.pkl")
    anomaly_det = load_model("models/anomaly_detector.pkl")
    tree = load_model("models/decision_tree.pkl")
    
    with open("models/model_info.json", 'r') as f:
        model_info = json.load(f)
    
    print("   ✅ All models loaded!")
    
    # =============================================
    # TEST 1: Activity Classification
    # =============================================
    print("\n" + "=" * 60)
    print("🏃 TEST 1: Activity Classification")
    print("=" * 60)
    
    test_cases_activity = [
        {"accel_x": 1.5, "accel_y": 0.8, "accel_z": 10.2, "pir": 1, "dist": 50,
         "desc": "Person moving near sensor"},
        {"accel_x": 0.01, "accel_y": 0.02, "accel_z": 9.8, "pir": 0, "dist": 350,
         "desc": "Empty room, no activity"},
        {"accel_x": 1.8, "accel_y": 1.9, "accel_z": 10.5, "pir": 1, "dist": 30,
         "desc": "Person walking quickly nearby"},
        {"accel_x": 0.1, "accel_y": 0.05, "accel_z": 9.8, "pir": 1, "dist": 80,
         "desc": "Person sitting still in room"},
    ]
    
    for i, tc in enumerate(test_cases_activity, 1):
        result = predict_activity(knn, tc["accel_x"], tc["accel_y"], tc["accel_z"], 
                                   tc["pir"], tc["dist"])
        print(f"\n   📌 Test {i}: {tc['desc']}")
        print(f"      Sensor: accel=({tc['accel_x']}, {tc['accel_y']}, {tc['accel_z']}), "
              f"PIR={tc['pir']}, Distance={tc['dist']}cm")
        print(f"      🔮 Predicted Activity: {result.upper()}")
    
    # =============================================
    # TEST 2: Anomaly Detection
    # =============================================
    print("\n" + "=" * 60)
    print("🔍 TEST 2: Anomaly Detection")
    print("=" * 60)
    
    test_cases_anomaly = [
        {"hour": 9, "pir": 1, "dist": 50, "entry": 1, "exit": 0, "office": 1,
         "desc": "Normal: Person enters at 9 AM"},
        {"hour": 23, "pir": 1, "dist": 30, "entry": 1, "exit": 0, "office": 0,
         "desc": "SUSPICIOUS: Motion at 11 PM!"},
        {"hour": 14, "pir": 0, "dist": 300, "entry": 0, "exit": 0, "office": 1,
         "desc": "Normal: Empty corridor at 2 PM"},
        {"hour": 2, "pir": 1, "dist": 15, "entry": 1, "exit": 0, "office": 0,
         "desc": "SUSPICIOUS: Someone at 2 AM!"},
        {"hour": 10, "pir": 1, "dist": 45, "entry": 0, "exit": 1, "office": 1,
         "desc": "Normal: Person leaving at 10 AM"},
    ]
    
    for i, tc in enumerate(test_cases_anomaly, 1):
        result = detect_anomaly(anomaly_det, tc["hour"], tc["pir"], tc["dist"],
                                tc["entry"], tc["exit"], tc["office"])
        print(f"\n   📌 Test {i}: {tc['desc']}")
        print(f"      Sensor: hour={tc['hour']}, PIR={tc['pir']}, dist={tc['dist']}cm, "
              f"office_hours={tc['office']}")
        print(f"      🔮 Result: {result}")
    
    # =============================================
    # TEST 3: Attendance Prediction
    # =============================================
    print("\n" + "=" * 60)
    print("📅 TEST 3: Attendance Prediction")
    print("=" * 60)
    
    test_cases_attend = [
        {"dow": 0, "dept": 1, "att_rate": 0.95, "late_rate": 0.05, "day": 15, "month": 2,
         "name": "Rahim Ahmed", "desc": "CSE student, 95% attendance, Monday"},
        {"dow": 3, "dept": 3, "att_rate": 0.50, "late_rate": 0.40, "day": 20, "month": 3,
         "name": "Low Attendance Student", "desc": "BBA student, only 50% attendance, Thursday"},
        {"dow": 2, "dept": 2, "att_rate": 0.88, "late_rate": 0.15, "day": 10, "month": 1,
         "name": "Shakib Rahman", "desc": "EEE student, 88% attendance, Wednesday"},
        {"dow": 0, "dept": 1, "att_rate": 0.30, "late_rate": 0.60, "day": 25, "month": 5,
         "name": "Irregular Student", "desc": "CSE student, 30% attendance, often late"},
    ]
    
    for i, tc in enumerate(test_cases_attend, 1):
        result = predict_attendance(tree, tc["dow"], tc["dept"], tc["att_rate"],
                                     tc["late_rate"], tc["day"], tc["month"])
        print(f"\n   📌 Test {i}: {tc['name']} — {tc['desc']}")
        print(f"      Features: day={tc['dow']}, dept={tc['dept']}, att_rate={tc['att_rate']}, "
              f"late_rate={tc['late_rate']}")
        print(f"      🔮 Prediction: {result}")
    
    # =============================================
    # LIVE SENSOR SIMULATION
    # =============================================
    print("\n" + "=" * 60)
    print("📡 LIVE SIMULATION — Full System Flow")
    print("=" * 60)
    
    print("""
    Simulating a complete attendance event:
    
    ⏰ Time: 8:45 AM (Office hours)
    📡 PIR Sensor → Motion DETECTED
    📏 Ultrasonic → Distance: 35cm (person nearby)
    🔄 Accelerometer → (1.2, 0.5, 10.1) — walking
    💳 RFID → Card scanned: A1B2C3D4
    """)
    
    # Step 1: Classify activity
    activity = predict_activity(knn, 1.2, 0.5, 10.1, 1, 35)
    print(f"    Step 1 — Activity: {activity.upper()}")
    
    # Step 2: Check for anomaly
    anomaly = detect_anomaly(anomaly_det, 8, 1, 35, 1, 0, 1)
    print(f"    Step 2 — Security: {anomaly}")
    
    # Step 3: Predict attendance likelihood
    attend = predict_attendance(tree, 0, 1, 0.95, 0.05, 24, 5)
    print(f"    Step 3 — Prediction: {attend}")
    
    print(f"""
    ✅ RESULT:
    ┌───────────────────────────────────────────┐
    │  User: Rahim Ahmed (STU-001)              │
    │  Activity: {activity.upper():>15}               │
    │  Security: {anomaly:>15}            │
    │  Attendance: MARKED ✅                     │
    │  Time: 08:45:23                           │
    │  Zone: Classroom A                        │
    └───────────────────────────────────────────┘
    """)
    
    print("=" * 60)
    print("🎉 All predictions complete!")
    print("=" * 60)
