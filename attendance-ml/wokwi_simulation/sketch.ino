#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define PIR_PIN    27
#define TRIG_PIN   13
#define ECHO_PIN   12
#define LED_GREEN  2
#define LED_RED    4
#define BUZZER_PIN 15
#define BTN1       14
#define BTN2       25
#define BTN3       26
#define BTN4       33

#define SCREEN_W 128
#define SCREEN_H 64
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);

struct Student {
  int pin;
  const char* name;
  const char* id;
  const char* dept;
  bool attended;  // <-- NEW: already attended today?
};

Student students[] = {
  {BTN1, "Rahim Ahmed",   "STU-001", "CSE", false},
  {BTN2, "Fatima Khan",   "STU-002", "CSE", false},
  {BTN3, "Karim Hossain", "STU-003", "EEE", false},
};

int attendanceCount = 0;
bool waitingForFinger = false;
unsigned long motionTime = 0;

float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long d = pulseIn(ECHO_PIN, HIGH, 30000);
  return d * 0.034 / 2.0;
}

void showReady() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(5, 0);
  display.println("=== READY ===");
  display.setCursor(5, 16);
  display.println("Waiting for motion..");
  display.setCursor(5, 32);
  display.print("Attendance: ");
  display.print(attendanceCount);
  display.print("/3");
  display.setCursor(5, 48);
  display.println("WiFi: Connected");
  display.display();
}

void showScanFinger() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(15, 5);
  display.println("PLACE");
  display.setCursor(10, 30);
  display.println("FINGER");
  display.setTextSize(1);
  display.setCursor(15, 52);
  display.println("on sensor now...");
  display.display();
}

void showWelcome(Student* s) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("=== WELCOME! ===");
  display.setTextSize(2);
  display.setCursor(0, 14);
  display.println(s->name);
  display.setTextSize(1);
  display.setCursor(0, 38);
  display.print("ID: ");
  display.println(s->id);
  display.setCursor(0, 50);
  display.print("Dept: ");
  display.print(s->dept);
  display.print(" #");
  display.println(attendanceCount);
  display.display();
}

void showAlreadyMarked(Student* s) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("!! ALREADY MARKED !!");
  display.setTextSize(2);
  display.setCursor(0, 16);
  display.println(s->name);
  display.setTextSize(1);
  display.setCursor(0, 40);
  display.println("You already gave");
  display.println("attendance today!");
  display.display();
}

void showRejected() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(10, 10);
  display.println("ACCESS");
  display.setCursor(10, 30);
  display.println("DENIED!");
  display.setTextSize(1);
  display.setCursor(5, 52);
  display.println("Unknown fingerprint!");
  display.display();
}

void beepOK() {
  tone(BUZZER_PIN, 1000, 150);
  delay(200);
  tone(BUZZER_PIN, 1500, 150);
  delay(200);
  noTone(BUZZER_PIN);
}

void beepAlready() {
  tone(BUZZER_PIN, 800, 100);
  delay(150);
  tone(BUZZER_PIN, 800, 100);
  delay(150);
  noTone(BUZZER_PIN);
}

void beepFail() {
  tone(BUZZER_PIN, 300, 400);
  delay(500);
  noTone(BUZZER_PIN);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== Attendance System Starting ===\n");

  pinMode(PIR_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BTN1, INPUT_PULLUP);
  pinMode(BTN2, INPUT_PULLUP);
  pinMode(BTN3, INPUT_PULLUP);
  pinMode(BTN4, INPUT_PULLUP);

  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(10, 10);
  display.println("ATTENDANCE SYSTEM");
  display.setCursor(10, 30);
  display.println("Initializing...");
  display.display();
  delay(2000);

  WiFi.begin("Wokwi-GUEST", "");
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  showReady();
  Serial.println("System Ready!\n");
}

void loop() {
  int motion = digitalRead(PIR_PIN);

  if (motion == HIGH && !waitingForFinger) {
    float dist = readDistance();
    Serial.println("----------------------------------");
    Serial.println("STEP 1: PIR -> Motion Detected!");
    Serial.print("STEP 2: Distance: ");
    Serial.print(dist);
    Serial.println(" cm");
    Serial.println("STEP 3: OLED -> Place Finger");

    showScanFinger();
    waitingForFinger = true;
    motionTime = millis();
  }

  if (waitingForFinger) {
    if (millis() - motionTime > 10000) {
      Serial.println("Timeout - no finger\n");
      waitingForFinger = false;
      showReady();
      return;
    }

    // Check student buttons
    for (int i = 0; i < 3; i++) {
      if (digitalRead(students[i].pin) == LOW) {

        // CHECK: Already attended?
        if (students[i].attended) {
          Serial.print("STEP 4: ");
          Serial.print(students[i].name);
          Serial.println(" -> ALREADY MARKED!");
          Serial.println("   Duplicate attendance blocked!\n");

          digitalWrite(LED_RED, HIGH);
          showAlreadyMarked(&students[i]);
          beepAlready();
          delay(3000);
          digitalWrite(LED_RED, LOW);

          waitingForFinger = false;
          showReady();
          return;
        }

        // NEW attendance
        attendanceCount++;
        students[i].attended = true;  // Mark as attended

        Serial.print("STEP 4: Fingerprint MATCH! -> ");
        Serial.println(students[i].name);
        Serial.print("   ID: ");
        Serial.print(students[i].id);
        Serial.print(" | Dept: ");
        Serial.println(students[i].dept);
        Serial.print("   Attendance #");
        Serial.print(attendanceCount);
        Serial.println("/3");

        digitalWrite(LED_GREEN, HIGH);
        showWelcome(&students[i]);
        beepOK();
        delay(3000);
        digitalWrite(LED_GREEN, LOW);

        waitingForFinger = false;
        showReady();
        Serial.println("----------------------------------\n");
        return;
      }
    }

    // Unknown button
    if (digitalRead(BTN4) == LOW) {
      Serial.println("STEP 4: Fingerprint -> REJECTED!");
      Serial.println("   UNKNOWN person! Alert sent!\n");

      digitalWrite(LED_RED, HIGH);
      showRejected();
      beepFail();
      delay(3000);
      digitalWrite(LED_RED, LOW);

      waitingForFinger = false;
      showReady();
      return;
    }
  }

  delay(100);
}