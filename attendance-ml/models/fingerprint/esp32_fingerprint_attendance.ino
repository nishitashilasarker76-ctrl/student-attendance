// ============================================
// Fingerprint Attendance System — ESP32
// Sensor: R307 / AS608
// Library: Adafruit_Fingerprint
// ============================================

#include <Adafruit_Fingerprint.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>

// ---- Fingerprint Sensor on UART2 ----
HardwareSerial mySerial(2);  // UART2: RX=16, TX=17
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&mySerial);

// ---- OLED Display ----
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ---- Pins ----
#define LED_GREEN  2
#define LED_RED    4
#define BUZZER_PIN 15
#define PIR_PIN    27

// ---- User Database (matches fingerprint IDs) ----
struct User {
  int fingerID;
  const char* name;
  const char* studentID;
  const char* dept;
};

User users[] = {
  {1, "Rahim Ahmed",   "STU-001", "CSE"},
  {2, "Fatima Khan",   "STU-002", "CSE"},
  {3, "Karim Hossain", "STU-003", "EEE"},
  {4, "Nasrin Akter",  "STU-004", "CSE"},
  {5, "Shakib Rahman", "STU-005", "EEE"},
  {6, "Aisha Begum",   "STU-006", "BBA"},
  {7, "Imran Hasan",   "STU-007", "CSE"},
  {8, "Sumaiya Islam", "STU-008", "EEE"},
};
const int NUM_USERS = 8;

int attendanceCount = 0;

void setup() {
  Serial.begin(115200);
  
  // Pins
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(PIR_PIN, INPUT);
  
  // OLED
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(10, 10);
  display.println("Fingerprint System");
  display.setCursor(10, 25);
  display.println("Initializing...");
  display.display();
  
  // Fingerprint sensor
  mySerial.begin(57600, SERIAL_8N1, 16, 17);
  finger.begin(57600);
  
  if (finger.verifyPassword()) {
    Serial.println("Fingerprint sensor found!");
    finger.getTemplateCount();
    Serial.print("Templates stored: ");
    Serial.println(finger.templateCount);
  } else {
    Serial.println("Fingerprint sensor NOT found!");
    while (1);
  }
  
  showReadyScreen();
}

void loop() {
  // Wait for PIR motion first (someone approaching)
  if (digitalRead(PIR_PIN) == HIGH) {
    showScanScreen();
    
    // Try to get fingerprint
    int fingerID = getFingerprintID();
    
    if (fingerID >= 0) {
      // Find user in database
      User* user = findUser(fingerID);
      
      if (user != NULL) {
        markAttendance(user);
      } else {
        // ID found in sensor but not in our database
        rejectUnknown(fingerID);
      }
    }
    // If fingerID == -1, no finger detected (just motion)
    // If fingerID == -2, finger detected but no match
    
    delay(2000);
    showReadyScreen();
  }
  
  delay(100);
}

int getFingerprintID() {
  uint8_t p = finger.getImage();
  if (p != FINGERPRINT_OK) return -1;  // No finger
  
  p = finger.image2Tz();
  if (p != FINGERPRINT_OK) return -1;
  
  p = finger.fingerFastSearch();
  if (p != FINGERPRINT_OK) {
    // Finger detected but no match!
    Serial.println("No match found!");
    rejectFinger();
    return -2;
  }
  
  // Match found!
  Serial.print("Found ID #");
  Serial.print(finger.fingerID);
  Serial.print(" with confidence ");
  Serial.println(finger.confidence);
  
  return finger.fingerID;
}

User* findUser(int fingerID) {
  for (int i = 0; i < NUM_USERS; i++) {
    if (users[i].fingerID == fingerID) {
      return &users[i];
    }
  }
  return NULL;
}

void markAttendance(User* user) {
  attendanceCount++;
  
  Serial.print("ATTENDANCE: ");
  Serial.print(user->name);
  Serial.print(" (");
  Serial.print(user->studentID);
  Serial.println(")");
  
  // Green LED + happy buzzer
  digitalWrite(LED_GREEN, HIGH);
  tone(BUZZER_PIN, 1000, 200);
  
  // OLED
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("=== WELCOME ===");
  display.setTextSize(2);
  display.setCursor(0, 14);
  display.println(user->name);
  display.setTextSize(1);
  display.setCursor(0, 38);
  display.print("ID: ");
  display.println(user->studentID);
  display.setCursor(0, 50);
  display.print("Dept: ");
  display.print(user->dept);
  display.print(" #");
  display.println(attendanceCount);
  display.display();
  
  // TODO: Send to Firebase/Cloud via WiFi
  // sendToCloud(user->studentID, user->name);
  
  delay(3000);
  digitalWrite(LED_GREEN, LOW);
}

void rejectFinger() {
  digitalWrite(LED_RED, HIGH);
  tone(BUZZER_PIN, 300, 500);
  
  display.clearDisplay();
  display.setTextSize(2);
  display.setCursor(10, 10);
  display.println("ACCESS");
  display.println(" DENIED!");
  display.setTextSize(1);
  display.setCursor(0, 50);
  display.println("Finger not registered!");
  display.display();
  
  delay(2000);
  digitalWrite(LED_RED, LOW);
}

void rejectUnknown(int id) {
  rejectFinger();
  Serial.print("Sensor ID #");
  Serial.print(id);
  Serial.println(" not in user database!");
}

void showReadyScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(5, 0);
  display.println("ATTENDANCE SYSTEM");
  display.drawLine(0, 10, 128, 10, WHITE);
  display.setCursor(5, 16);
  display.println("Place finger on");
  display.setCursor(5, 28);
  display.println("sensor to scan...");
  display.setCursor(5, 44);
  display.print("Total today: ");
  display.println(attendanceCount);
  display.display();
}

void showScanScreen() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setCursor(10, 15);
  display.println("SCANNING");
  display.setCursor(30, 40);
  display.println("...");
  display.display();
}
