"""
╔══════════════════════════════════════════════════════════════╗
║  APPROACH 2: TensorFlow / Keras (Deep Learning)             ║
║──────────────────────────────────────────────────────────────║
║  Models: Dense Neural Network, LSTM, Autoencoder            ║
║  GPU ছাড়াই CPU তে চলবে! Simple architecture ব্যবহার করেছি   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import numpy as np
import pandas as pd
import json, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def print_header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ================================================================
#  LOAD DATA
# ================================================================
print_header("📂 LOADING DATA")

sensor_df = pd.read_csv("data/sensor_readings.csv")
attend_df = pd.read_csv("data/attendance_records.csv")
print(f"  Sensor : {sensor_df.shape}")
print(f"  Attend : {attend_df.shape}")

all_results = {}


# ================================================================
#  MODEL 1: Dense Neural Network — Activity Classification
# ================================================================
print_header("🧠 MODEL 1: Dense Neural Network (DNN) — Activity Classification")

# Prepare
feature_cols = ['accel_x', 'accel_y', 'accel_z', 'pir_motion', 'ultrasonic_cm']
X = sensor_df[feature_cols].values.astype(np.float32)

le = LabelEncoder()
y_encoded = le.fit_transform(sensor_df['activity_label'].values)
num_classes = len(le.classes_)
y_onehot = keras.utils.to_categorical(y_encoded, num_classes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42
)
y_test_labels = np.argmax(y_test, axis=1)

print(f"  Classes     : {list(le.classes_)}")
print(f"  Num classes : {num_classes}")
print(f"  Features    : {len(feature_cols)}")
print(f"  Train       : {X_train.shape[0]}")
print(f"  Test        : {X_test.shape[0]}")

# ---- Build DNN Model ----
print(f"\n  🔨 Building Dense Neural Network...")

dnn_model = keras.Sequential([
    layers.Input(shape=(len(feature_cols),)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
], name="Activity_DNN")

dnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

dnn_model.summary()

# ---- Train ----
print(f"\n  🚀 Training DNN...")
history_dnn = dnn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.15,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=0)
    ]
)

# ---- Evaluate ----
loss, accuracy = dnn_model.evaluate(X_test, y_test, verbose=0)
y_pred_dnn = np.argmax(dnn_model.predict(X_test, verbose=0), axis=1)

print(f"\n  📊 DNN Results:")
print(f"     Loss     : {loss:.4f}")
print(f"     Accuracy : {accuracy:.4f} ({accuracy:.2%})")
print(f"\n  Classification Report:")
print(classification_report(y_test_labels, y_pred_dnn, 
                            target_names=le.classes_, zero_division=0))

# Training history
final_epoch = len(history_dnn.history['accuracy'])
print(f"  📈 Training History:")
print(f"     Epochs trained  : {final_epoch}")
print(f"     Final train acc : {history_dnn.history['accuracy'][-1]:.4f}")
print(f"     Final val acc   : {history_dnn.history['val_accuracy'][-1]:.4f}")

all_results['dnn_activity'] = {
    'accuracy': round(accuracy, 4),
    'loss': round(loss, 4),
    'epochs': final_epoch,
    'architecture': '64→128→64→32→softmax',
}


# ================================================================
#  MODEL 2: Autoencoder — Anomaly Detection
# ================================================================
print_header("🔍 MODEL 2: Autoencoder Neural Network — Anomaly Detection")

print("""
  📖 Autoencoder কিভাবে কাজ করে:
     1. শুধুমাত্র NORMAL data দিয়ে train করে
     2. Model শেখে normal data কিভাবে দেখতে
     3. যখন anomaly data আসে, reconstruction error বেড়ে যায়
     4. High error = ANOMALY! 🚨
""")

# Prepare
anom_features = ['hour', 'pir_motion', 'ultrasonic_cm', 'ir_entry', 'ir_exit', 'is_office_hours']
X_anom = sensor_df[anom_features].values.astype(np.float32)
y_anom = sensor_df['is_anomaly'].astype(int).values

scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)

# Split: train ONLY on normal data
X_normal = X_anom_scaled[y_anom == 0]
X_anomaly = X_anom_scaled[y_anom == 1]

X_train_n, X_val_n = train_test_split(X_normal, test_size=0.2, random_state=42)

print(f"  Normal train : {X_train_n.shape[0]}")
print(f"  Normal val   : {X_val_n.shape[0]}")
print(f"  Anomaly test : {X_anomaly.shape[0]}")

# ---- Build Autoencoder ----
print(f"\n  🔨 Building Autoencoder...")

input_dim = len(anom_features)
encoding_dim = 3  # Bottleneck

# Encoder
encoder_input = layers.Input(shape=(input_dim,))
encoded = layers.Dense(16, activation='relu')(encoder_input)
encoded = layers.Dense(8, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(8, activation='relu')(encoded)
decoded = layers.Dense(16, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(encoder_input, decoded, name="Anomaly_Autoencoder")
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# ---- Train ----
print(f"\n  🚀 Training Autoencoder (on normal data only)...")
history_ae = autoencoder.fit(
    X_train_n, X_train_n,  # Input = Output (reconstruct itself)
    epochs=100,
    batch_size=32,
    validation_data=(X_val_n, X_val_n),
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# ---- Evaluate ----
# Calculate reconstruction error
recon_normal = autoencoder.predict(X_val_n, verbose=0)
recon_anomaly = autoencoder.predict(X_anomaly, verbose=0)

mse_normal = np.mean(np.square(X_val_n - recon_normal), axis=1)
mse_anomaly = np.mean(np.square(X_anomaly - recon_anomaly), axis=1)

# Set threshold at 95th percentile of normal errors
threshold = np.percentile(mse_normal, 95)

print(f"\n  📊 Autoencoder Results:")
print(f"     Normal MSE  (mean) : {mse_normal.mean():.6f}")
print(f"     Anomaly MSE (mean) : {mse_anomaly.mean():.6f}")
print(f"     Threshold (95th %) : {threshold:.6f}")
print(f"     Anomaly MSE / Normal MSE ratio: {mse_anomaly.mean() / mse_normal.mean():.2f}x")

# Classify
y_pred_normal = (mse_normal > threshold).astype(int)
y_pred_anomaly = (mse_anomaly > threshold).astype(int)

# Combined evaluation
y_true_all = np.concatenate([np.zeros(len(X_val_n)), np.ones(len(X_anomaly))])
y_pred_all = np.concatenate([y_pred_normal, y_pred_anomaly])

acc_ae = accuracy_score(y_true_all, y_pred_all)
print(f"     Combined Accuracy  : {acc_ae:.4f} ({acc_ae:.2%})")
print(f"\n  Classification Report:")
print(classification_report(y_true_all, y_pred_all, 
                            target_names=['Normal', 'Anomaly'], zero_division=0))

# Detection rates
normal_correct = (y_pred_normal == 0).sum() / len(y_pred_normal)
anomaly_detected = (y_pred_anomaly == 1).sum() / len(y_pred_anomaly)
print(f"  📈 Detection Rates:")
print(f"     Normal correctly identified : {normal_correct:.2%}")
print(f"     Anomalies detected          : {anomaly_detected:.2%}")

all_results['autoencoder_anomaly'] = {
    'accuracy': round(acc_ae, 4),
    'threshold': round(float(threshold), 6),
    'normal_mse': round(float(mse_normal.mean()), 6),
    'anomaly_mse': round(float(mse_anomaly.mean()), 6),
    'anomaly_detection_rate': round(float(anomaly_detected), 4),
    'architecture': 'encoder(16→8→3) + decoder(8→16→6)',
}


# ================================================================
#  MODEL 3: Simple RNN/Dense — Attendance Prediction
# ================================================================
print_header("📅 MODEL 3: Neural Network — Attendance Prediction")

# Prepare
attend_clean = attend_df[attend_df['status'].isin(['present', 'absent'])].copy()
attend_clean['date_parsed'] = pd.to_datetime(attend_clean['date'])
attend_clean['day_of_week'] = attend_clean['date_parsed'].dt.dayofweek
attend_clean['day'] = attend_clean['date_parsed'].dt.day
attend_clean['month'] = attend_clean['date_parsed'].dt.month

dept_map = {"CSE": 1, "EEE": 2, "BBA": 3}
attend_clean['dept_code'] = attend_clean['department'].map(dept_map).fillna(0)

user_stats = attend_clean.groupby('user_uid').agg(
    total=('status', 'count'),
    present_count=('status', lambda x: (x == 'present').sum()),
    late_count=('is_late', lambda x: x.astype(int).sum()),
).reset_index()
user_stats['attendance_rate'] = user_stats['present_count'] / user_stats['total']
user_stats['late_rate'] = user_stats['late_count'] / user_stats['present_count'].clip(lower=1)

attend_clean = attend_clean.merge(user_stats[['user_uid', 'attendance_rate', 'late_rate']], on='user_uid')

att_features = ['day_of_week', 'dept_code', 'attendance_rate', 'late_rate', 'day', 'month']
X_att = attend_clean[att_features].values.astype(np.float32)
y_att = (attend_clean['status'] == 'present').astype(int).values

scaler_att = StandardScaler()
X_att_scaled = scaler_att.fit_transform(X_att)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_att_scaled, y_att, test_size=0.2, random_state=42, stratify=y_att
)

print(f"  Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

# ---- Build Model ----
print(f"\n  🔨 Building Attendance Prediction NN...")

att_model = keras.Sequential([
    layers.Input(shape=(len(att_features),)),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary: present/absent
], name="Attendance_NN")

att_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

att_model.summary()

# ---- Train ----
# Handle class imbalance
class_counts = np.bincount(y_tr)
total = len(y_tr)
class_weight = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
print(f"\n  Class weights: {class_weight}")

print(f"  🚀 Training Attendance NN...")
history_att = att_model.fit(
    X_tr, y_tr,
    epochs=80,
    batch_size=16,
    validation_split=0.15,
    class_weight=class_weight,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# ---- Evaluate ----
loss_att, acc_att = att_model.evaluate(X_te, y_te, verbose=0)
y_pred_att = (att_model.predict(X_te, verbose=0) > 0.5).astype(int).flatten()

print(f"\n  📊 Attendance NN Results:")
print(f"     Loss     : {loss_att:.4f}")
print(f"     Accuracy : {acc_att:.4f} ({acc_att:.2%})")
print(f"\n  Classification Report:")
print(classification_report(y_te, y_pred_att, target_names=['Absent', 'Present'], zero_division=0))

all_results['nn_attendance'] = {
    'accuracy': round(acc_att, 4),
    'loss': round(loss_att, 4),
    'architecture': '32→64→32→16→sigmoid',
}


# ================================================================
#  SAVE MODELS
# ================================================================
print_header("💾 SAVING TENSORFLOW MODELS")

os.makedirs("models/tensorflow", exist_ok=True)

dnn_model.save("models/tensorflow/activity_dnn.keras")
print("  ✅ Activity DNN → models/tensorflow/activity_dnn.keras")

autoencoder.save("models/tensorflow/anomaly_autoencoder.keras")
print("  ✅ Autoencoder → models/tensorflow/anomaly_autoencoder.keras")

att_model.save("models/tensorflow/attendance_nn.keras")
print("  ✅ Attendance NN → models/tensorflow/attendance_nn.keras")

with open("models/tensorflow/results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print("  ✅ Results → models/tensorflow/results.json")


# ================================================================
#  FINAL SUMMARY
# ================================================================
print_header("🏆 TENSORFLOW / KERAS — FINAL RESULTS")

print(f"""
  ┌──────────────────────────────┬────────────┬─────────────────────────────┐
  │  Model                       │  Accuracy  │  Architecture               │
  ├──────────────────────────────┼────────────┼─────────────────────────────┤
  │  🧠 DNN Activity Classifier  │  {all_results['dnn_activity']['accuracy']:.4f}    │  {all_results['dnn_activity']['architecture']:>27} │
  │  🔍 Autoencoder Anomaly Det  │  {all_results['autoencoder_anomaly']['accuracy']:.4f}    │  {all_results['autoencoder_anomaly']['architecture']:>27} │
  │  📅 NN Attendance Predictor  │  {all_results['nn_attendance']['accuracy']:.4f}    │  {all_results['nn_attendance']['architecture']:>27} │
  └──────────────────────────────┴────────────┴─────────────────────────────┘

  ✅ Approach 2 (TensorFlow/Keras) — COMPLETE!
  ➡️  Next: Run approach3_tflite_edge.py for Edge ML on ESP32!
""")
