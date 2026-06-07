"""
=============================================================
STEP 2: Train ML Models for Attendance System
=============================================================
3টি Model Train করব:

  Model 1: Activity Classification (walking/sitting/standing/running/idle)
           → Accelerometer data থেকে user কি করছে detect করবে
  
  Model 2: Anomaly Detection (normal vs anomaly)
           → Unusual activity detect করবে (unauthorized access, off-hours)
  
  Model 3: Attendance Prediction
           → Historical data থেকে predict করবে কে আসবে/আসবে না
           
All using simple Python with scikit-learn!
=============================================================
"""

import csv
import os
import json
import pickle
from collections import Counter

# ============================================
# Simple ML Library (NO external dependency!)
# ============================================
# আমরা scikit-learn ছাড়াই basic ML implement করব
# যাতে কোনো installation issue না হয়

import math
import random

random.seed(42)


def load_csv(filepath):
    """Load CSV file as list of dicts"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================
# MODEL 1: Activity Classification
# Using K-Nearest Neighbors (from scratch!)
# ============================================
class SimpleKNN:
    """K-Nearest Neighbors — built from scratch!"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        print(f"   ✅ KNN trained with {len(X)} samples, k={self.k}")
    
    def _distance(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def predict_one(self, x):
        distances = [(self._distance(x, x_train), y) 
                     for x_train, y in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda d: d[0])
        k_nearest = [d[1] for d in distances[:self.k]]
        # Majority vote
        counter = Counter(k_nearest)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for p, actual in zip(predictions, y) if p == actual)
        return correct / len(y)


# ============================================
# MODEL 2: Anomaly Detection
# Using simple threshold + statistical method
# ============================================
class SimpleAnomalyDetector:
    """Anomaly detection using Z-score method"""
    
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.means = []
        self.stds = []
    
    def fit(self, X_normal):
        """Train on NORMAL data only"""
        n_features = len(X_normal[0])
        self.means = []
        self.stds = []
        
        for i in range(n_features):
            values = [x[i] for x in X_normal]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            self.means.append(mean)
            self.stds.append(max(std, 0.001))  # Avoid division by zero
        
        print(f"   ✅ Anomaly Detector trained on {len(X_normal)} normal samples")
    
    def predict_one(self, x):
        z_scores = [abs((x[i] - self.means[i]) / self.stds[i]) 
                    for i in range(len(x))]
        avg_z = sum(z_scores) / len(z_scores)
        return 1 if avg_z > self.threshold else 0  # 1 = anomaly
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for p, actual in zip(predictions, y) if p == actual)
        return correct / len(y)


# ============================================
# MODEL 3: Attendance Prediction
# Using Decision Tree (from scratch!)
# ============================================
class SimpleDecisionTree:
    """Simple Decision Tree for attendance prediction"""
    
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def _gini(self, groups, classes):
        total = sum(len(g) for g in groups)
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for c in classes:
                p = [row[-1] for row in group].count(c) / size
                score += p * p
            gini += (1.0 - score) * (size / total)
        return gini
    
    def _split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    def _best_split(self, dataset):
        classes = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        
        n_features = len(dataset[0]) - 1
        for index in range(n_features):
            for row in dataset:
                groups = self._split(index, row[index], dataset)
                gini = self._gini(groups, classes)
                if gini < best_score:
                    best_index = index
                    best_value = row[index]
                    best_score = gini
                    best_groups = groups
        
        return {'index': best_index, 'value': best_value, 'groups': best_groups}
    
    def _terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
    
    def _split_node(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        
        if not left or not right:
            node['left'] = node['right'] = self._terminal(left + right)
            return
        
        if depth >= self.max_depth:
            node['left'] = self._terminal(left)
            node['right'] = self._terminal(right)
            return
        
        if len(left) <= 1:
            node['left'] = self._terminal(left)
        else:
            node['left'] = self._best_split(left)
            self._split_node(node['left'], depth + 1)
        
        if len(right) <= 1:
            node['right'] = self._terminal(right)
        else:
            node['right'] = self._best_split(right)
            self._split_node(node['right'], depth + 1)
    
    def fit(self, X, y):
        dataset = [list(x) + [label] for x, label in zip(X, y)]
        # Sample if too large (for speed)
        if len(dataset) > 500:
            dataset = random.sample(dataset, 500)
        self.tree = self._best_split(dataset)
        self._split_node(self.tree, 1)
        print(f"   ✅ Decision Tree trained with max_depth={self.max_depth}")
    
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
    
    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for p, actual in zip(predictions, y) if p == actual)
        return correct / len(y)


# ============================================
# DATA PREPARATION
# ============================================
def prepare_activity_data(sensor_data):
    """Prepare data for activity classification"""
    X, y = [], []
    for row in sensor_data:
        features = [
            float(row['accel_x']),
            float(row['accel_y']),
            float(row['accel_z']),
            float(row['pir_motion']),
            float(row['ultrasonic_cm']),
        ]
        label = row['activity_label']
        X.append(features)
        y.append(label)
    return X, y


def prepare_anomaly_data(sensor_data):
    """Prepare data for anomaly detection"""
    X, y = [], []
    for row in sensor_data:
        features = [
            float(row['hour']),
            float(row['pir_motion']),
            float(row['ultrasonic_cm']),
            float(row['ir_entry']),
            float(row['ir_exit']),
            float(row['is_office_hours']),
        ]
        label = int(row['is_anomaly'])
        X.append(features)
        y.append(label)
    return X, y


def prepare_attendance_data(attendance_data):
    """Prepare data for attendance prediction"""
    X, y = [], []
    
    # Create user attendance history features
    user_history = {}
    for row in attendance_data:
        uid = row['user_uid']
        if uid not in user_history:
            user_history[uid] = {'present': 0, 'absent': 0, 'late': 0}
        if row['status'] == 'present':
            user_history[uid]['present'] += 1
            if int(row['is_late']):
                user_history[uid]['late'] += 1
        elif row['status'] == 'absent':
            user_history[uid]['absent'] += 1
    
    for row in attendance_data:
        if row['status'] == 'unauthorized':
            continue
        
        uid = row['user_uid']
        hist = user_history.get(uid, {'present': 0, 'absent': 0, 'late': 0})
        total = hist['present'] + hist['absent']
        if total == 0:
            continue
        
        # Day of week (0=Monday ... 6=Sunday)
        from datetime import datetime
        date = datetime.strptime(row['date'], "%Y-%m-%d")
        day_of_week = date.weekday()
        
        # Department encoding
        dept_map = {"CSE": 1, "EEE": 2, "BBA": 3, "UNKNOWN": 0}
        dept_code = dept_map.get(row['department'], 0)
        
        features = [
            day_of_week,
            dept_code,
            hist['present'] / total,          # Attendance rate
            hist['late'] / max(hist['present'], 1),  # Late rate
            date.day,                          # Day of month
            date.month,                        # Month
        ]
        label = 1 if row['status'] == 'present' else 0
        X.append(features)
        y.append(label)
    
    return X, y


def train_test_split(X, y, test_ratio=0.2):
    """Split data into train and test sets"""
    combined = list(zip(X, y))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_ratio))
    train = combined[:split_idx]
    test = combined[split_idx:]
    X_train = [t[0] for t in train]
    y_train = [t[1] for t in train]
    X_test = [t[0] for t in test]
    y_test = [t[1] for t in test]
    return X_train, X_test, y_train, y_test


def confusion_matrix(y_true, y_pred, labels):
    """Calculate confusion matrix"""
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
    return matrix


def print_confusion_matrix(matrix, labels):
    """Pretty print confusion matrix"""
    # Header
    header = f"{'':>12}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    print("-" * len(header))
    for actual in labels:
        row = f"{actual:>12}"
        for pred in labels:
            row += f"{matrix[actual][pred]:>10}"
        print(row)


def classification_report(y_true, y_pred, labels):
    """Generate classification report"""
    print(f"\n{'Label':>12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 55)
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{str(label):>12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")


# ============================================
# MAIN TRAINING PIPELINE
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 STEP 2: Training ML Models")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    sensor_data = load_csv("data/sensor_readings.csv")
    attendance_data = load_csv("data/attendance_records.csv")
    print(f"   Sensor readings : {len(sensor_data)}")
    print(f"   Attendance records: {len(attendance_data)}")
    
    results = {}
    
    # =============================================
    # MODEL 1: Activity Classification (KNN)
    # =============================================
    print("\n" + "=" * 60)
    print("🏃 MODEL 1: Activity Classification (K-Nearest Neighbors)")
    print("=" * 60)
    
    X_act, y_act = prepare_activity_data(sensor_data)
    X_train, X_test, y_train, y_test = train_test_split(X_act, y_act, 0.2)
    
    print(f"\n   Training samples : {len(X_train)}")
    print(f"   Testing samples  : {len(X_test)}")
    print(f"   Features         : accel_x, accel_y, accel_z, pir_motion, ultrasonic_cm")
    print(f"   Classes          : {list(set(y_act))}")
    
    # Train
    knn = SimpleKNN(k=5)
    knn.fit(X_train, y_train)
    
    # Evaluate
    accuracy = knn.score(X_test, y_test)
    print(f"\n   📊 Accuracy: {accuracy:.2%}")
    results['activity_classification'] = {'accuracy': round(accuracy, 4), 'model': 'KNN (k=5)'}
    
    # Classification report
    y_pred = knn.predict(X_test)
    labels_act = sorted(list(set(y_act)))
    print("\n   📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels_act)
    print_confusion_matrix(cm, labels_act)
    classification_report(y_test, y_pred, labels_act)
    
    # =============================================
    # MODEL 2: Anomaly Detection
    # =============================================
    print("\n" + "=" * 60)
    print("🔍 MODEL 2: Anomaly Detection (Z-Score Method)")
    print("=" * 60)
    
    X_anom, y_anom = prepare_anomaly_data(sensor_data)
    
    # Split: train on normal data only
    X_normal = [x for x, y in zip(X_anom, y_anom) if y == 0]
    X_anomaly = [x for x, y in zip(X_anom, y_anom) if y == 1]
    
    print(f"\n   Normal samples  : {len(X_normal)}")
    print(f"   Anomaly samples : {len(X_anomaly)}")
    print(f"   Features        : hour, pir_motion, ultrasonic_cm, ir_entry, ir_exit, is_office_hours")
    
    # Train on normal data
    detector = SimpleAnomalyDetector(threshold=1.8)
    detector.fit(X_normal[:int(len(X_normal)*0.8)])  # 80% for training
    
    # Test on mix of normal + anomaly
    X_test_anom = X_normal[int(len(X_normal)*0.8):] + X_anomaly
    y_test_anom = [0] * len(X_normal[int(len(X_normal)*0.8):]) + [1] * len(X_anomaly)
    
    accuracy_anom = detector.score(X_test_anom, y_test_anom)
    print(f"\n   📊 Accuracy: {accuracy_anom:.2%}")
    results['anomaly_detection'] = {'accuracy': round(accuracy_anom, 4), 'model': 'Z-Score Anomaly Detector'}
    
    y_pred_anom = detector.predict(X_test_anom)
    labels_anom = [0, 1]
    print("\n   📋 Confusion Matrix (0=Normal, 1=Anomaly):")
    cm_anom = confusion_matrix(y_test_anom, y_pred_anom, labels_anom)
    print_confusion_matrix(cm_anom, labels_anom)
    classification_report(y_test_anom, y_pred_anom, labels_anom)
    
    # =============================================
    # MODEL 3: Attendance Prediction (Decision Tree)
    # =============================================
    print("\n" + "=" * 60)
    print("📅 MODEL 3: Attendance Prediction (Decision Tree)")
    print("=" * 60)
    
    X_att, y_att = prepare_attendance_data(attendance_data)
    X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(X_att, y_att, 0.2)
    
    print(f"\n   Training samples : {len(X_train_att)}")
    print(f"   Testing samples  : {len(X_test_att)}")
    print(f"   Features         : day_of_week, dept, attendance_rate, late_rate, day, month")
    print(f"   Classes          : 0=Absent, 1=Present")
    
    # Train
    tree = SimpleDecisionTree(max_depth=4)
    tree.fit(X_train_att, y_train_att)
    
    # Evaluate
    accuracy_att = tree.score(X_test_att, y_test_att)
    print(f"\n   📊 Accuracy: {accuracy_att:.2%}")
    results['attendance_prediction'] = {'accuracy': round(accuracy_att, 4), 'model': 'Decision Tree (depth=4)'}
    
    y_pred_att = tree.predict(X_test_att)
    labels_att = [0, 1]
    print("\n   📋 Confusion Matrix (0=Absent, 1=Present):")
    cm_att = confusion_matrix(y_test_att, y_pred_att, labels_att)
    print_confusion_matrix(cm_att, labels_att)
    classification_report(y_test_att, y_pred_att, labels_att)
    
    # =============================================
    # SAVE MODELS
    # =============================================
    print("\n" + "=" * 60)
    print("💾 Saving Models...")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    
    # Save model parameters as JSON (portable!)
    model_info = {
        "models": {
            "activity_classifier": {
                "type": "KNN",
                "k": 5,
                "accuracy": results['activity_classification']['accuracy'],
                "features": ["accel_x", "accel_y", "accel_z", "pir_motion", "ultrasonic_cm"],
                "classes": labels_act,
                "training_samples": len(X_train),
            },
            "anomaly_detector": {
                "type": "Z-Score",
                "threshold": 1.8,
                "accuracy": results['anomaly_detection']['accuracy'],
                "features": ["hour", "pir_motion", "ultrasonic_cm", "ir_entry", "ir_exit", "is_office_hours"],
                "means": detector.means,
                "stds": detector.stds,
                "training_samples": len(X_normal),
            },
            "attendance_predictor": {
                "type": "Decision Tree",
                "max_depth": 4,
                "accuracy": results['attendance_prediction']['accuracy'],
                "features": ["day_of_week", "department", "attendance_rate", "late_rate", "day", "month"],
                "classes": ["absent", "present"],
                "training_samples": len(X_train_att),
            }
        },
        "metadata": {
            "generated_by": "Attendance ML Pipeline",
            "total_sensor_data": len(sensor_data),
            "total_attendance_data": len(attendance_data),
        }
    }
    
    with open("models/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    print("   ✅ Model info → models/model_info.json")
    
    # Save models using pickle
    with open("models/knn_activity.pkl", 'wb') as f:
        pickle.dump(knn, f)
    print("   ✅ KNN model → models/knn_activity.pkl")
    
    with open("models/anomaly_detector.pkl", 'wb') as f:
        pickle.dump(detector, f)
    print("   ✅ Anomaly detector → models/anomaly_detector.pkl")
    
    with open("models/decision_tree.pkl", 'wb') as f:
        pickle.dump(tree, f)
    print("   ✅ Decision tree → models/decision_tree.pkl")
    
    # =============================================
    # FINAL SUMMARY
    # =============================================
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print(f"""
    ┌─────────────────────────────────────────────────────┐
    │  Model                          │  Accuracy         │
    ├─────────────────────────────────┼───────────────────┤
    │  🏃 Activity Classification     │  {results['activity_classification']['accuracy']:.2%}           │
    │  🔍 Anomaly Detection           │  {results['anomaly_detection']['accuracy']:.2%}           │
    │  📅 Attendance Prediction       │  {results['attendance_prediction']['accuracy']:.2%}           │
    └─────────────────────────────────┴───────────────────┘
    """)
    print("✅ All models saved to ./models/ folder!")
    print("➡️  Next: Run step3_predict.py to make predictions!")
