"""
╔══════════════════════════════════════════════════════════════╗
║  APPROACH 1: Scikit-learn (Industry Standard ML)            ║
║──────────────────────────────────────────────────────────────║
║  Models: Random Forest, SVM, Logistic Regression,           ║
║          Gradient Boosting, Isolation Forest                 ║
║  সবচেয়ে popular এবং production-ready approach!              ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import json, os, pickle, warnings
warnings.filterwarnings('ignore')

# ---- Scikit-learn imports ----
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              IsolationForest)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def print_header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_results(name, y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  📊 {name}")
    print(f"     Accuracy : {acc:.4f} ({acc:.2%})")
    print(f"     F1-Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\n     Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print(f"     Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return acc


# ================================================================
#  LOAD DATA
# ================================================================
print_header("📂 LOADING DATA")

sensor_df = pd.read_csv("data/sensor_readings.csv")
attend_df = pd.read_csv("data/attendance_records.csv")

print(f"  Sensor data  : {sensor_df.shape[0]} rows × {sensor_df.shape[1]} cols")
print(f"  Attend data  : {attend_df.shape[0]} rows × {attend_df.shape[1]} cols")

# Quick data preview
print(f"\n  Sensor columns: {list(sensor_df.columns)}")
print(f"  Activity distribution:\n{sensor_df['activity_label'].value_counts().to_string()}")

all_results = {}


# ================================================================
#  MODEL 1: Activity Classification (Multi-class)
# ================================================================
print_header("🏃 MODEL 1: Activity Classification — 5 Algorithms Compared!")

# Prepare features
feature_cols = ['accel_x', 'accel_y', 'accel_z', 'pir_motion', 'ultrasonic_cm']
X = sensor_df[feature_cols].values
y_raw = sensor_df['activity_label'].values

# Encode labels
le_activity = LabelEncoder()
y = le_activity.fit_transform(y_raw)
class_names = list(le_activity.classes_)
print(f"\n  Classes: {class_names}")
print(f"  Encoded: {list(range(len(class_names)))}")

# Scale features
scaler_act = StandardScaler()
X_scaled = scaler_act.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ---- Train 5 different models ----
models_act = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    "SVM (RBF kernel)": SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, multi_class='multinomial', random_state=42),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
}

act_scores = {}
best_act_acc = 0
best_act_model = None
best_act_name = ""

for name, model in models_act.items():
    print(f"\n  🔄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = print_results(name, y_test, y_pred, class_names)
    act_scores[name] = acc
    
    if acc > best_act_acc:
        best_act_acc = acc
        best_act_model = model
        best_act_name = name

# Cross-validation for best model
cv_scores = cross_val_score(best_act_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"\n  🏆 BEST MODEL: {best_act_name}")
print(f"     Test Accuracy     : {best_act_acc:.4f}")
print(f"     5-Fold CV Mean    : {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

all_results['activity_classification'] = {
    'scores': act_scores,
    'best_model': best_act_name,
    'best_accuracy': best_act_acc,
    'cv_mean': round(cv_scores.mean(), 4),
}

# Feature importance (if Random Forest or Gradient Boosting)
if hasattr(best_act_model, 'feature_importances_'):
    importance = best_act_model.feature_importances_
    print(f"\n  📊 Feature Importance:")
    for fname, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"     {fname:>15}: {imp:.4f} {bar}")


# ================================================================
#  MODEL 2: Anomaly Detection
# ================================================================
print_header("🔍 MODEL 2: Anomaly Detection — Isolation Forest + Supervised")

# Prepare features
anom_features = ['hour', 'pir_motion', 'ultrasonic_cm', 'ir_entry', 'ir_exit', 'is_office_hours']
X_anom = sensor_df[anom_features].values
y_anom = sensor_df['is_anomaly'].astype(int).values

scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)

print(f"  Normal : {sum(y_anom == 0)}")
print(f"  Anomaly: {sum(y_anom == 1)}")

# ---- Method A: Isolation Forest (Unsupervised) ----
print(f"\n  🔄 Training Isolation Forest (unsupervised)...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_anom_scaled)
y_pred_iso = iso_forest.predict(X_anom_scaled)
y_pred_iso = np.array([1 if p == -1 else 0 for p in y_pred_iso])  # -1 = anomaly

acc_iso = accuracy_score(y_anom, y_pred_iso)
print(f"  📊 Isolation Forest Accuracy: {acc_iso:.4f}")
print(classification_report(y_anom, y_pred_iso, target_names=['Normal', 'Anomaly'], zero_division=0))

# ---- Method B: Random Forest (Supervised) ----
print(f"  🔄 Training Random Forest (supervised)...")
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X_anom_scaled, y_anom, test_size=0.2, random_state=42, stratify=y_anom
)

rf_anom = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_anom.fit(X_tr_a, y_tr_a)
y_pred_rf_a = rf_anom.predict(X_te_a)
acc_rf_anom = print_results("Random Forest (Supervised)", y_te_a, y_pred_rf_a, ['Normal', 'Anomaly'])

# Feature importance
importance_anom = rf_anom.feature_importances_
print(f"\n  📊 Anomaly Feature Importance:")
for fname, imp in sorted(zip(anom_features, importance_anom), key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"     {fname:>18}: {imp:.4f} {bar}")

all_results['anomaly_detection'] = {
    'isolation_forest': round(acc_iso, 4),
    'random_forest_supervised': round(acc_rf_anom, 4),
    'best_model': 'Random Forest (Supervised)',
    'best_accuracy': round(acc_rf_anom, 4),
}


# ================================================================
#  MODEL 3: Attendance Prediction
# ================================================================
print_header("📅 MODEL 3: Attendance Prediction — Multiple Classifiers")

# Prepare features
attend_clean = attend_df[attend_df['status'].isin(['present', 'absent'])].copy()
attend_clean['date_parsed'] = pd.to_datetime(attend_clean['date'])
attend_clean['day_of_week'] = attend_clean['date_parsed'].dt.dayofweek
attend_clean['day'] = attend_clean['date_parsed'].dt.day
attend_clean['month'] = attend_clean['date_parsed'].dt.month

# Department encoding
dept_map = {"CSE": 1, "EEE": 2, "BBA": 3}
attend_clean['dept_code'] = attend_clean['department'].map(dept_map).fillna(0)

# Compute per-user stats
user_stats = attend_clean.groupby('user_uid').agg(
    total=('status', 'count'),
    present_count=('status', lambda x: (x == 'present').sum()),
    late_count=('is_late', lambda x: x.astype(int).sum()),
).reset_index()
user_stats['attendance_rate'] = user_stats['present_count'] / user_stats['total']
user_stats['late_rate'] = user_stats['late_count'] / user_stats['present_count'].clip(lower=1)

attend_clean = attend_clean.merge(user_stats[['user_uid', 'attendance_rate', 'late_rate']], on='user_uid')

att_features = ['day_of_week', 'dept_code', 'attendance_rate', 'late_rate', 'day', 'month']
X_att = attend_clean[att_features].values
y_att = (attend_clean['status'] == 'present').astype(int).values

scaler_att = StandardScaler()
X_att_scaled = scaler_att.fit_transform(X_att)

X_tr_att, X_te_att, y_tr_att, y_te_att = train_test_split(
    X_att_scaled, y_att, test_size=0.2, random_state=42, stratify=y_att
)
print(f"  Train: {X_tr_att.shape[0]} | Test: {X_te_att.shape[0]}")

# Train multiple models
models_att = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    "SVM": SVC(kernel='rbf', C=10, class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
}

att_scores = {}
best_att_acc = 0
best_att_name = ""

for name, model in models_att.items():
    print(f"\n  🔄 Training {name}...")
    model.fit(X_tr_att, y_tr_att)
    y_pred_att = model.predict(X_te_att)
    acc = print_results(name, y_te_att, y_pred_att, ['Absent', 'Present'])
    att_scores[name] = acc
    if acc > best_att_acc:
        best_att_acc = acc
        best_att_name = name

print(f"\n  🏆 BEST MODEL: {best_att_name} → {best_att_acc:.4f}")

all_results['attendance_prediction'] = {
    'scores': att_scores,
    'best_model': best_att_name,
    'best_accuracy': best_att_acc,
}


# ================================================================
#  SAVE RESULTS
# ================================================================
print_header("💾 SAVING MODELS & RESULTS")

os.makedirs("models/sklearn", exist_ok=True)

with open("models/sklearn/results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print("  ✅ Results → models/sklearn/results.json")

# Save best models
pickle.dump(best_act_model, open("models/sklearn/best_activity_model.pkl", 'wb'))
pickle.dump(rf_anom, open("models/sklearn/best_anomaly_model.pkl", 'wb'))
pickle.dump(scaler_act, open("models/sklearn/scaler_activity.pkl", 'wb'))
pickle.dump(scaler_anom, open("models/sklearn/scaler_anomaly.pkl", 'wb'))
pickle.dump(le_activity, open("models/sklearn/label_encoder.pkl", 'wb'))
print("  ✅ Models saved to models/sklearn/")


# ================================================================
#  FINAL COMPARISON TABLE
# ================================================================
print_header("🏆 FINAL COMPARISON — All Models All Tasks")

print(f"""
  ┌───────────────────────────┬──────────────────────────────────────────────────┐
  │  TASK                     │  Model Accuracies                                │
  ├───────────────────────────┼──────────────────────────────────────────────────┤
  │                           │                                                  │""")
print(f"  │  🏃 Activity Classification│", end="")
for name, score in act_scores.items():
    print(f"  {name[:15]:>15}: {score:.3f}", end="")
print(f"│")
print(f"  │                           │  🏆 Best: {best_act_name} ({best_act_acc:.3f})           │")
print(f"  ├───────────────────────────┼──────────────────────────────────────────────────┤")
print(f"  │  🔍 Anomaly Detection     │  Isolation Forest: {all_results['anomaly_detection']['isolation_forest']:.3f}                       │")
print(f"  │                           │  Random Forest:    {all_results['anomaly_detection']['random_forest_supervised']:.3f}                       │")
print(f"  ├───────────────────────────┼──────────────────────────────────────────────────┤")
print(f"  │  📅 Attendance Prediction │", end="")
for name, score in att_scores.items():
    print(f"  {name[:15]:>15}: {score:.3f}", end="")
print(f"│")
print(f"  │                           │  🏆 Best: {best_att_name} ({best_att_acc:.3f})           │")
print(f"  └───────────────────────────┴──────────────────────────────────────────────────┘")

print(f"\n  ✅ Approach 1 (Scikit-learn) — COMPLETE!")
print(f"  ➡️  Next: Run approach2_tensorflow.py for Deep Learning!")
