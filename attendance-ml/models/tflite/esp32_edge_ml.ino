// ============================================
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
const float scaler_mean[] = {-0.0003f, 0.0006f, 9.7567f, 0.4902f, 180.2192f};
const float scaler_std[]  = {0.4631f, 0.4415f, 0.3540f, 0.4999f, 130.0357f};

// ---- Class Names ----
const char* class_names[] = {"idle", "running", "sitting", "standing", "walking"}; 
const int num_classes = 5;

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

void setup() {
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
}

// ---- Read sensors & predict ----
void loop() {
  // Read accelerometer (simulated / MPU6050)
  float accel_x = random(-200, 200) / 100.0;  // Replace with real sensor
  float accel_y = random(-200, 200) / 100.0;
  float accel_z = random(900, 1100) / 100.0;
  
  // Read PIR
  float pir = digitalRead(PIR_PIN);
  
  // Read Ultrasonic
  float distance = readUltrasonic();
  
  // ---- Normalize using scaler parameters ----
  float features[5];
  features[0] = (accel_x - scaler_mean[0]) / scaler_std[0];
  features[1] = (accel_y - scaler_mean[1]) / scaler_std[1];
  features[2] = (accel_z - scaler_mean[2]) / scaler_std[2];
  features[3] = (pir - scaler_mean[3]) / scaler_std[3];
  features[4] = (distance - scaler_mean[4]) / scaler_std[4];
  
  // ---- Set input tensor ----
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = features[i];
  }
  
  // ---- Run inference! ----
  interpreter->Invoke();
  
  // ---- Get prediction ----
  float max_score = 0;
  int predicted_class = 0;
  for (int i = 0; i < num_classes; i++) {
    float score = output->data.f[i];
    if (score > max_score) {
      max_score = score;
      predicted_class = i;
    }
  }
  
  // ---- Print result ----
  Serial.print("🔮 Activity: ");
  Serial.print(class_names[predicted_class]);
  Serial.print(" (confidence: ");
  Serial.print(max_score * 100, 1);
  Serial.println("%)");
  
  delay(1000);
}

float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 30000);
  return dur * 0.034 / 2.0;
}
