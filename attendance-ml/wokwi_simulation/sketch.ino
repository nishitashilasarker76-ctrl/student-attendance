#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ===== PINS =====
#define PIR_PIN    27
#define TRIG_PIN   13
#define ECHO_PIN   12
#define LED_GREEN  2
#define LED_RED    4
#define BUZZER_PIN 15
#define RFID_SS    5
#define RFID_RST   0

#define SCREEN_W 128
#define SCREEN_H 64
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);
MFRC522 rfid(RFID_SS, RFID_RST);

const char* FIREBASE_URL = "https://smart-attendance-and-activity-default-rtdb.asia-southeast1.firebasedatabase.app";

// ===== STUDENT DATABASE — matched by RFID card UID =====
struct Student {
  const char* uid;    // RFID card unique ID
  const char* name;
  const char* id;
  const char* dept;
  bool attended;
};

// Wokwi generates random UIDs for each virtual card
// We'll match ANY scanned card to students in order
// First unique card = Rahim, Second = Fatima, Third = Karim
// After that = UNKNOWN

Student students[] = {
  {"", "Rahim Ahmed",   "STU-001", "CSE", false},
  {"", "Fatima Khan",   "STU-002", "CSE", false},
  {"", "Karim Hossain", "STU-003", "EEE", false},
};
int NUM_STUDENTS = 3;
int registeredCount = 0;

int attendanceCount = 0;
bool waitingForCard = false;
unsigned long motionTime = 0;

// ===== ACTIVITY (auto-random) =====
const char* activities[] = {"WALKING","WALK_UPSTAIRS","WALK_DOWNSTAIRS","SITTING","STANDING","LAYING"};
float accelData[][3] = {
  {0.28,-0.02,-0.07},{0.26,-0.03,-0.12},{0.31,-0.02,-0.05},
  {0.27,-0.01,-0.10},{0.28,-0.02,-0.09},{0.28,-0.03,-0.02}
};
int currentActivity = 0;
unsigned long lastActChange = 0;

const char* detectActivity() {
  if (millis() - lastActChange > random(2000, 6000)) {
    currentActivity = random(0, 6);
    lastActChange = millis();
  }
  return activities[currentActivity];
}

void getAccel(float &ax, float &ay, float &az) {
  ax = accelData[currentActivity][0] + random(-50,50)/1000.0;
  ay = accelData[currentActivity][1] + random(-50,50)/1000.0;
  az = accelData[currentActivity][2] + random(-50,50)/1000.0;
}

float readDistance() {
  digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  return pulseIn(ECHO_PIN, HIGH, 30000) * 0.034 / 2.0;
}

// ===== GET CARD UID AS STRING =====
String getCardUID() {
  String uid = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) uid += "0";
    uid += String(rfid.uid.uidByte[i], HEX);
  }
  uid.toUpperCase();
  return uid;
}

// ===== FIND STUDENT BY UID =====
int findStudent(String uid) {
  // Check if this UID is already registered
  for (int i = 0; i < NUM_STUDENTS; i++) {
    if (String(students[i].uid) == uid) {
      return i;  // Found!
    }
  }
  return -1;  // Not found
}

// ===== AUTO-REGISTER NEW CARD =====
int autoRegister(String uid) {
  if (registeredCount >= NUM_STUDENTS) {
    return -1;  // All slots full = UNKNOWN person
  }
  // Assign this card to next available student
  students[registeredCount].uid = strdup(uid.c_str());
  Serial.print("   Auto-registered: Card ");
  Serial.print(uid);
  Serial.print(" -> ");
  Serial.println(students[registeredCount].name);
  registeredCount++;
  return registeredCount - 1;
}

// ===== FIREBASE =====
void sendToFirebase(Student* s, float dist, const char* activity) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(FIREBASE_URL) + "/attendance/" + String(s->id) + ".json";
  float ax,ay,az; getAccel(ax,ay,az);
  
  String json = "{";
  json += "\"name\":\"" + String(s->name) + "\",";
  json += "\"student_id\":\"" + String(s->id) + "\",";
  json += "\"department\":\"" + String(s->dept) + "\",";
  json += "\"rfid_uid\":\"" + String(s->uid) + "\",";
  json += "\"status\":\"present\",";
  json += "\"distance_cm\":" + String((int)dist) + ",";
  json += "\"activity\":\"" + String(activity) + "\",";
  json += "\"accel_x\":" + String(ax,3) + ",";
  json += "\"accel_y\":" + String(ay,3) + ",";
  json += "\"accel_z\":" + String(az,3) + ",";
  json += "\"attendance_num\":" + String(attendanceCount);
  json += "}";

  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.PUT(json);
  Serial.println(code == 200 ? "   Firebase: SAVED!" : "   Firebase: Error");
  http.end();
}

void sendAlertFirebase(String uid, float dist, const char* activity) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(FIREBASE_URL) + "/alerts/latest.json";
  String json = "{\"type\":\"UNAUTHORIZED\",\"rfid_uid\":\"" + uid + "\",";
  json += "\"distance\":" + String((int)dist) + ",\"activity\":\"" + String(activity) + "\",";
  json += "\"message\":\"Unknown RFID card detected!\"}";
  http.begin(url); http.addHeader("Content-Type","application/json");
  http.PUT(json); http.end();
}

// ===== DISPLAYS =====
void showLiveSensors() {
  float dist = readDistance();
  int pir = digitalRead(PIR_PIN);
  const char* act = detectActivity();
  float ax,ay,az; getAccel(ax,ay,az);

  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0);  display.println("== LIVE SENSORS ==");
  display.drawLine(0,10,128,10,WHITE);
  display.setCursor(0,13); display.print("PIR: "); display.println(pir?"MOTION!":"No motion");
  display.setCursor(0,23); display.print("Dist: "); display.print((int)dist); display.println(" cm");
  display.setCursor(0,33); display.print("Acc:"); display.print(ax,2); display.print(","); display.print(ay,2); display.print(","); display.println(az,2);
  display.setCursor(0,43); display.print(">> "); display.print(act); display.println(" <<");
  display.setCursor(0,55); display.print(attendanceCount); display.print("/3|Reg:"); display.print(registeredCount); display.print("|WiFi");
  display.display();
}

void showScanCard() {
  float dist = readDistance();
  const char* act = detectActivity();
  float ax,ay,az; getAccel(ax,ay,az);

  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0);  display.println("!! MOTION DETECTED !!");
  display.drawLine(0,10,128,10,WHITE);
  display.setCursor(0,14); display.print("Dist:"); display.print((int)dist); display.print("cm Act:"); display.println(act);
  display.setCursor(0,26); display.print("Acc:"); display.print(ax,2); display.print(","); display.print(ay,2); display.print(","); display.println(az,2);
  display.drawLine(0,38,128,38,WHITE);
  display.setTextSize(2);
  display.setCursor(10,42); display.println("TAP CARD");
  display.display();

  Serial.println("----------------------------------");
  Serial.print("Motion! Dist:"); Serial.print((int)dist); Serial.print("cm Act:"); Serial.println(act);
}

void showWelcome(Student* s, float dist, const char* act) {
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0);  display.println("=== WELCOME! ===");
  display.setTextSize(2);
  display.setCursor(0,11); display.println(s->name);
  display.setTextSize(1);
  display.setCursor(0,32); display.print(s->id); display.print("|"); display.println(s->dept);
  display.setCursor(0,42); display.print("UID:"); display.println(s->uid);
  display.setCursor(0,52); display.print("Act:"); display.print(act); display.print(" #"); display.print(attendanceCount); display.print("/3");
  display.display();
}

void showAlready(Student* s) {
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0);  display.println("!! ALREADY MARKED !!");
  display.setTextSize(2);
  display.setCursor(0,16); display.println(s->name);
  display.setTextSize(1);
  display.setCursor(0,42); display.print("UID:"); display.println(s->uid);
  display.setCursor(0,52); display.println("Already recorded!");
  display.display();
}

void showRejected(String uid) {
  display.clearDisplay();
  display.setTextSize(2); display.setTextColor(WHITE);
  display.setCursor(10,3);  display.println("ACCESS");
  display.setCursor(10,23); display.println("DENIED!");
  display.setTextSize(1);
  display.setCursor(0,45); display.print("UID:"); display.println(uid);
  display.setCursor(0,55); display.println("Unknown card! ALERT!");
  display.display();
}

void beepOK() { tone(BUZZER_PIN,1000,150); delay(200); tone(BUZZER_PIN,1500,150); delay(200); noTone(BUZZER_PIN); }
void beepAlready() { tone(BUZZER_PIN,800,100); delay(150); tone(BUZZER_PIN,800,100); delay(150); noTone(BUZZER_PIN); }
void beepFail() { tone(BUZZER_PIN,300,400); delay(500); noTone(BUZZER_PIN); }

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(0));

  Serial.println("\n========================================");
  Serial.println("  ATTENDANCE + ACTIVITY TRACKING");
  Serial.println("  RFID Card Authentication");
  Serial.println("  6 Activities (UCI HAR)");
  Serial.println("========================================\n");

  pinMode(PIR_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // OLED
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(5,5);  display.println("ATTENDANCE SYSTEM");
  display.setCursor(5,18); display.println("RFID + Activity Track");
  display.drawLine(0,28,128,28,WHITE);
  display.setCursor(5,32); display.println("Tap 1st card = Rahim");
  display.setCursor(5,42); display.println("Tap 2nd card = Fatima");
  display.setCursor(5,52); display.println("Tap 3rd card = Karim");
  display.display();
  delay(3000);

  // RFID
  SPI.begin();
  rfid.PCD_Init();
  Serial.println("RFID Reader: Ready");
  Serial.println("Tap different cards to register students!\n");

  // WiFi
  display.clearDisplay();
  display.setCursor(5,20); display.println("Connecting WiFi...");
  display.display();

  WiFi.begin("Wokwi-GUEST", "");
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  Serial.println("WiFi Connected!");

  display.setCursor(5,35); display.println("WiFi: OK!");
  display.setCursor(5,48); display.println("Tap RFID card...");
  display.display();
  delay(1500);
}

// ===== MAIN LOOP =====
void loop() {
  int motion = digitalRead(PIR_PIN);

  if (!waitingForCard) {
    showLiveSensors();
    delay(400);
  }

  // PIR triggers scan mode
  if (motion == HIGH && !waitingForCard) {
    showScanCard();
    waitingForCard = true;
    motionTime = millis();
  }

  // Waiting for RFID card tap
  if (waitingForCard) {
    if (millis() - motionTime > 15000) {
      Serial.println("Timeout\n");
      waitingForCard = false;
      return;
    }

    // Check for new RFID card
    if (!rfid.PICC_IsNewCardPresent()) return;
    if (!rfid.PICC_ReadCardSerial()) return;

    String uid = getCardUID();
    float dist = readDistance();
    const char* activity = detectActivity();

    Serial.print("\nCard detected! UID: ");
    Serial.println(uid);

    // Find or register student
    int idx = findStudent(uid);
    
    if (idx == -1) {
      // New card — try to auto-register
      idx = autoRegister(uid);
    }

    if (idx >= 0) {
      // KNOWN STUDENT
      if (students[idx].attended) {
        // Already marked today
        Serial.print(students[idx].name);
        Serial.println(" -> ALREADY MARKED!");
        
        digitalWrite(LED_RED, HIGH);
        showAlready(&students[idx]);
        beepAlready();
        delay(3000);
        digitalWrite(LED_RED, LOW);
      } else {
        // NEW ATTENDANCE!
        attendanceCount++;
        students[idx].attended = true;

        Serial.print("MATCH: "); Serial.print(students[idx].name);
        Serial.print(" ("); Serial.print(students[idx].id);
        Serial.print(") | UID: "); Serial.print(uid);
        Serial.print(" | Act: "); Serial.println(activity);

        digitalWrite(LED_GREEN, HIGH);
        showWelcome(&students[idx], dist, activity);
        beepOK();
        sendToFirebase(&students[idx], dist, activity);
        delay(3000);
        digitalWrite(LED_GREEN, LOW);
      }
    } else {
      // UNKNOWN — all 3 slots full, this is 4th+ card
      Serial.print("REJECTED! Unknown UID: ");
      Serial.println(uid);

      digitalWrite(LED_RED, HIGH);
      showRejected(uid);
      beepFail();
      sendAlertFirebase(uid, dist, activity);
      delay(3000);
      digitalWrite(LED_RED, LOW);
    }

    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
    waitingForCard = false;
    Serial.println("----------------------------------\n");
  }

  delay(100);
}