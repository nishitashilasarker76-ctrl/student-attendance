"""
╔══════════════════════════════════════════════════════════════╗
║  APPROACH 3: TensorFlow Lite — ESP32 Edge Deployment        ║
║──────────────────────────────────────────────────────────────║
║  Train → Convert to TFLite → Generate C Header → ESP32 run! ║
║  Model সরাসরি microcontroller এ চলবে — cloud দরকার নেই!     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import json, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def print_header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ================================================================
#  LOAD DATA
# ================================================================
print_header("📂 LOADING DATA")

sensor_df = pd.read_csv("data/sensor_readings.csv")
print(f"  Sensor data: {sensor_df.shape}")

all_results = {}


# ================================================================
#  STEP 1: Train a TINY model (optimized for microcontroller)
# ================================================================
print_header("🤏 STEP 1: Train Tiny Model (ESP32 optimized)")

print("""
  📖 Edge ML এর নিয়ম:
     • Model খুব ছোট হতে হবে (< 100KB)
     • কম memory ব্যবহার করতে হবে (ESP32 has 520KB RAM)
     • শুধু Dense layers, কোনো complex architecture নয়
     • INT8 quantization করে size আরো কমানো যায়
""")

# Prepare data
feature_cols = ['accel_x', 'accel_y', 'accel_z', 'pir_motion', 'ultrasonic_cm']
X = sensor_df[feature_cols].values.astype(np.float32)

le = LabelEncoder()
y = le.fit_transform(sensor_df['activity_label'].values)
num_classes = len(le.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Input features : {len(feature_cols)}")
print(f"  Output classes : {num_classes} → {list(le.classes_)}")
print(f"  Train/Test     : {len(X_train)}/{len(X_test)}")

# ---- Build TINY model ----
print(f"\n  🔨 Building Tiny Edge Model...")

tiny_model = keras.Sequential([
    layers.Input(shape=(len(feature_cols),)),
    layers.Dense(16, activation='relu'),   # Very small!
    layers.Dense(8, activation='relu'),    # Bottleneck
    layers.Dense(num_classes, activation='softmax')
], name="TinyEdge_Activity")

tiny_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tiny_model.summary()

# Count parameters
total_params = tiny_model.count_params()
print(f"\n  📐 Total parameters: {total_params}")
print(f"  📐 Estimated size: ~{total_params * 4 / 1024:.1f} KB (float32)")
print(f"  📐 After INT8 quant: ~{total_params / 1024:.1f} KB")

# ---- Train ----
print(f"\n  🚀 Training Tiny Model...")
history = tiny_model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_split=0.15, verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
)

loss, acc = tiny_model.evaluate(X_test, y_test, verbose=0)
print(f"\n  📊 Tiny Model Accuracy: {acc:.4f} ({acc:.2%})")
print(f"  📊 Loss: {loss:.4f}")

all_results['tiny_model'] = {
    'accuracy': round(acc, 4),
    'params': total_params,
    'architecture': '16→8→softmax',
}


# ================================================================
#  STEP 2: Convert to TFLite
# ================================================================
print_header("🔄 STEP 2: Convert to TensorFlow Lite")

os.makedirs("models/tflite", exist_ok=True)

# ---- Method A: Float32 (no quantization) ----
print(f"  Converting to TFLite (Float32)...")
converter_f32 = tf.lite.TFLiteConverter.from_keras_model(tiny_model)
tflite_f32 = converter_f32.convert()

f32_path = "models/tflite/activity_model_f32.tflite"
with open(f32_path, 'wb') as f:
    f.write(tflite_f32)
f32_size = os.path.getsize(f32_path)
print(f"  ✅ Float32 model: {f32_path} ({f32_size} bytes = {f32_size/1024:.1f} KB)")

# ---- Method B: INT8 Quantization (smallest!) ----
print(f"\n  Converting to TFLite (INT8 Quantized)...")
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(tiny_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for quantization calibration
def representative_dataset():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1]]

converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

try:
    tflite_int8 = converter_int8.convert()
    int8_path = "models/tflite/activity_model_int8.tflite"
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    int8_size = os.path.getsize(int8_path)
    print(f"  ✅ INT8 model: {int8_path} ({int8_size} bytes = {int8_size/1024:.1f} KB)")
    print(f"  📉 Size reduction: {(1 - int8_size/f32_size)*100:.1f}% smaller!")
except Exception as e:
    print(f"  ⚠️ INT8 quantization issue: {e}")
    print(f"  Using Float16 instead...")
    converter_f16 = tf.lite.TFLiteConverter.from_keras_model(tiny_model)
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter_f16.convert()
    int8_path = "models/tflite/activity_model_f16.tflite"
    with open(int8_path, 'wb') as f:
        f.write(tflite_f16)
    int8_size = os.path.getsize(int8_path)
    print(f"  ✅ Float16 model: {int8_path} ({int8_size} bytes = {int8_size/1024:.1f} KB)")
    print(f"  📉 Size reduction: {(1 - int8_size/f32_size)*100:.1f}% smaller!")


# ---- Verify TFLite accuracy ----
print(f"\n  🔍 Verifying TFLite model accuracy...")
interpreter = tf.lite.Interpreter(model_path=f32_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input shape : {input_details[0]['shape']}")
print(f"  Output shape: {output_details[0]['shape']}")

correct = 0
total = min(200, len(X_test))
for i in range(total):
    interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1].astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)
    if pred == y_test[i]:
        correct += 1

tflite_acc = correct / total
print(f"  TFLite Accuracy: {tflite_acc:.4f} ({tflite_acc:.2%})")
print(f"  Original Keras : {acc:.4f}")
print(f"  Accuracy drop  : {abs(acc - tflite_acc)*100:.2f}%")


# ================================================================
#  STEP 3: Generate C Header File for ESP32
# ================================================================
print_header("📄 STEP 3: Generate C Header for ESP32 Arduino")

def convert_to_c_array(tflite_model_bytes, var_name="activity_model"):
    """Convert TFLite model to C byte array for Arduino"""
    c_array = ", ".join([f"0x{b:02x}" for b in tflite_model_bytes])
    
    header = f"""// ==========================================
// Auto-generated TFLite model for ESP32
// Model: Activity Classification
// Size: {len(tflite_model_bytes)} bytes
// ==========================================

#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

const unsigned char {var_name}_tflite[] = {{
  {c_array}
}};

const unsigned int {var_name}_tflite_len = {len(tflite_model_bytes)};

// Input: {len(feature_cols)} features [{', '.join(feature_cols)}]
// Output: {num_classes} classes [{', '.join(le.classes_)}]
// 
// Scaler means: [{', '.join(f'{m:.4f}' for m in scaler.mean_)}]
// Scaler stds:  [{', '.join(f'{s:.4f}' for s in scaler.scale_)}]

#endif // {var_name.upper()}_H
"""
    return header


c_header = convert_to_c_array(tflite_f32, "activity_model")

header_path = "models/tflite/activity_model.h"
with open(header_path, 'w') as f:
    f.write(c_header)
print(f"  ✅ C Header: {header_path}")
print(f"  📐 Array size: {len(tflite_f32)} bytes")


# ================================================================
#  STEP 4: Generate ESP32 Arduino Code
# ================================================================
print_header("📝 STEP 4: ESP32 Arduino Code (uses the TFLite model)")

esp32_code = f'''// ============================================
// ESP32 Edge ML — Activity Classification
// TensorFlow Lite Micro on ESP32
// ============================================
// 
// এই code ESP32 তে সরাসরি ML model run করে!
// Cloud connection দরকার নেই!
//
// Install Libraries:
//   - TensorFlowLite_ESP32 (Arduino Library Manager)
//   - Adafruit_SSD1306
// ============================================

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the converted model
#include "activity_model.h"

// ---- Scaler Parameters (from training) ----
const float scaler_mean[] = {{{', '.join(f'{m:.4f}f' for m in scaler.mean_)}}};
const float scaler_std[]  = {{{', '.join(f'{s:.4f}f' for s in scaler.scale_)}}};

// ---- Class Names ----
const char* class_names[] = {{"{('", "'.join(le.classes_))}"}}; 
const int num_classes = {num_classes};

// ---- TFLite Variables ----
constexpr int kTensorArenaSize = 8 * 1024;  // 8KB arena
uint8_t tensor_arena[kTensorArenaSize];

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ---- Sensor Pins ----
#define PIR_PIN       27
#define TRIG_PIN      13
#define ECHO_PIN      12

void setup() {{
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // Load TFLite model
  model = tflite::GetModel(activity_model_tflite);
  
  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("✅ Edge ML Model loaded on ESP32!");
  Serial.print("   Model size: ");
  Serial.print(activity_model_tflite_len);
  Serial.println(" bytes");
}}

// ---- Read sensors & predict ----
void loop() {{
  // Read accelerometer (simulated / MPU6050)
  float accel_x = random(-200, 200) / 100.0;  // Replace with real sensor
  float accel_y = random(-200, 200) / 100.0;
  float accel_z = random(900, 1100) / 100.0;
  
  // Read PIR
  float pir = digitalRead(PIR_PIN);
  
  // Read Ultrasonic
  float distance = readUltrasonic();
  
  // ---- Normalize using scaler parameters ----
  float features[{len(feature_cols)}];
  features[0] = (accel_x - scaler_mean[0]) / scaler_std[0];
  features[1] = (accel_y - scaler_mean[1]) / scaler_std[1];
  features[2] = (accel_z - scaler_mean[2]) / scaler_std[2];
  features[3] = (pir - scaler_mean[3]) / scaler_std[3];
  features[4] = (distance - scaler_mean[4]) / scaler_std[4];
  
  // ---- Set input tensor ----
  for (int i = 0; i < {len(feature_cols)}; i++) {{
    input->data.f[i] = features[i];
  }}
  
  // ---- Run inference! ----
  interpreter->Invoke();
  
  // ---- Get prediction ----
  float max_score = 0;
  int predicted_class = 0;
  for (int i = 0; i < num_classes; i++) {{
    float score = output->data.f[i];
    if (score > max_score) {{
      max_score = score;
      predicted_class = i;
    }}
  }}
  
  // ---- Print result ----
  Serial.print("🔮 Activity: ");
  Serial.print(class_names[predicted_class]);
  Serial.print(" (confidence: ");
  Serial.print(max_score * 100, 1);
  Serial.println("%)");
  
  delay(1000);
}}

float readUltrasonic() {{
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 30000);
  return dur * 0.034 / 2.0;
}}
'''

esp32_path = "models/tflite/esp32_edge_ml.ino"
with open(esp32_path, 'w') as f:
    f.write(esp32_code)
print(f"  ✅ ESP32 Arduino code: {esp32_path}")


# ================================================================
#  STEP 5: Model Size Comparison
# ================================================================
print_header("📐 STEP 5: Model Size Comparison for ESP32")

print(f"""
  ┌───────────────────────────────┬──────────┬──────────────────────────────┐
  │  Model Format                 │  Size    │  ESP32 Compatible?           │
  ├───────────────────────────────┼──────────┼──────────────────────────────┤
  │  Keras (.keras)               │  ~50 KB  │  ❌ Too large, needs Python  │
  │  TFLite Float32 (.tflite)     │  {f32_size/1024:>5.1f} KB │  ✅ Yes! With TFLite Micro    │
  │  TFLite INT8/F16 (.tflite)    │  {int8_size/1024:>5.1f} KB │  ✅ Best! Smallest & fastest  │
  │  C Header Array (.h)          │  ~{f32_size*3/1024:>4.0f} KB │  ✅ Embedded directly in code │
  ├───────────────────────────────┼──────────┼──────────────────────────────┤
  │  ESP32 Flash Memory           │  4 MB    │  ✅ Plenty of room!          │
  │  ESP32 RAM                    │  520 KB  │  ✅ Model + 8KB arena        │
  └───────────────────────────────┴──────────┴──────────────────────────────┘

  📊 Accuracy Comparison:
     Original Keras  : {acc:.4f}
     TFLite Float32  : {tflite_acc:.4f}
     Accuracy loss   : {abs(acc - tflite_acc)*100:.2f}% (negligible!)
""")

all_results['tflite'] = {
    'f32_size_bytes': f32_size,
    'int8_size_bytes': int8_size,
    'keras_accuracy': round(acc, 4),
    'tflite_accuracy': round(tflite_acc, 4),
}

with open("models/tflite/results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print_header("🎉 APPROACH 3 COMPLETE — Edge ML Ready for ESP32!")
print(f"""
  Files generated:
  ├── models/tflite/activity_model_f32.tflite    ← TFLite model (Float32)
  ├── models/tflite/activity_model_int8.tflite   ← TFLite model (INT8/F16) 
  ├── models/tflite/activity_model.h             ← C header for Arduino
  ├── models/tflite/esp32_edge_ml.ino            ← Complete ESP32 code
  └── models/tflite/results.json                 ← Results & metrics

  🚀 Deployment Steps:
     1. Copy activity_model.h + esp32_edge_ml.ino to Arduino IDE
     2. Install TensorFlowLite_ESP32 library
     3. Upload to ESP32
     4. Model runs ON the chip — no internet needed! ⚡
""")
