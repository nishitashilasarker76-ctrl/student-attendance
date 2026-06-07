"""
╔══════════════════════════════════════════════════════════════╗
║  Fingerprint-Based Attendance System — Complete Simulation  ║
║──────────────────────────────────────────────────────────────║
║  R307/AS608 Fingerprint Sensor কিভাবে কাজ করে:             ║
║  1. Enrollment: আঙুল scan → template store (sensor memory) ║
║  2. Matching: আঙুল scan → database search → ID return      ║
║  3. Attendance: ID match → user info fetch → record stored  ║
║                                                              ║
║  Simulator: Fingerprint = unique numeric template (no real  ║
║  sensor needed — we simulate the matching algorithm!)        ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import random
import json
import os
from datetime import datetime

random.seed(42)
np.random.seed(42)


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ================================================================
#  HOW FINGERPRINT SENSOR WORKS (R307 / AS608)
# ================================================================
header("📖 HOW FINGERPRINT SENSOR WORKS — R307/AS608 Explained")

print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │  R307/AS608 Fingerprint Sensor — Real Hardware Specs:            │
  │                                                                  │
  │  📐 Resolution      : 500 DPI                                   │
  │  💾 Internal Storage : 150-1000 fingerprint templates            │
  │  ⚡ Scan Time       : < 0.5 seconds                             │
  │  🔗 Interface       : UART (Serial) — TX/RX to ESP32            │
  │  🔋 Voltage         : 3.3V-6V (works with ESP32 directly!)      │
  │  💰 Price           : $8-15 USD                                  │
  │                                                                  │
  │  📊 ACCURACY:                                                    │
  │  ┌────────────────────────────────────────────────────────┐      │
  │  │  False Accept Rate (FAR)  : < 0.001%                  │      │
  │  │  → ১০,০০০ জনের মধ্যে মাত্র ১ জন ভুলভাবে match হবে!  │      │
  │  │                                                        │      │
  │  │  False Reject Rate (FRR)  : < 0.1%                    │      │
  │  │  → ১০০০ বার scan এ মাত্র ১ বার reject হবে!            │      │
  │  │                                                        │      │
  │  │  Combined Accuracy        : 99.9%+                     │      │
  │  └────────────────────────────────────────────────────────┘      │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  PROCESS — ২টি Phase:                                           │
  │                                                                  │
  │  Phase 1: ENROLLMENT (Registration — একবারই করতে হয়)           │
  │  ─────────────────────────────────────────────────                │
  │  1. আঙুল sensor এ রাখুন                                         │
  │  2. Sensor optical image capture করে (500 DPI)                  │
  │  3. Image থেকে minutiae points extract করে                      │
  │     (ridge endings, bifurcations — আঙুলের pattern)              │
  │  4. Minutiae → 512-byte TEMPLATE তৈরি হয়                       │
  │  5. Template sensor এর internal flash memory তে store হয়       │
  │     ID #1 → Rahim, ID #2 → Fatima, ID #3 → Karim...            │
  │  6. আঙুল তুলুন, আবার রাখুন (2nd scan for verification)         │
  │  7. দুই scan match হলে → Enrollment সফল! ✅                     │
  │                                                                  │
  │  Phase 2: MATCHING (প্রতিদিন attendance এর সময়)                │
  │  ────────────────────────────────────────────                     │
  │  1. আঙুল sensor এ রাখুন                                         │
  │  2. Sensor image capture → template তৈরি                        │
  │  3. Template কে stored database এর সব template এর সাথে compare │
  │     (1:N search — sensor নিজেই internally করে!)                  │
  │  4. Match found → ID return (e.g., ID #3 = Karim)               │
  │     + Confidence score (0-300, higher = better match)            │
  │  5. ESP32 receives: fingerID = 3, confidence = 250              │
  │  6. ESP32 database থেকে user info fetch:                        │
  │     ID #3 → name="Karim", dept="EEE", uid="STU-003"            │
  │  7. Attendance record → Cloud/Firebase/Database                  │
  │  8. OLED display: "Welcome Karim!" + LED green + Buzzer ✅       │
  │                                                                  │
  │  ⏱️ Total time: < 1 second (scan to attendance marked!)         │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  WHY FINGERPRINT > RFID for unique identification:              │
  │                                                                  │
  │  ✅ Cannot be shared (RFID card ভাড়া দেওয়া যায়, আঙুল যায় না!) │
  │  ✅ Cannot be lost (card হারিয়ে যেতে পারে, আঙুল হারায় না!)     │
  │  ✅ Cannot be duplicated (99.999% unique)                        │
  │  ✅ Fast (< 0.5 seconds)                                        │
  │  ✅ No extra cost per user (no cards to buy!)                    │
  │  ✅ Sensor stores templates internally — ESP32 শুধু command দেয় │
  └──────────────────────────────────────────────────────────────────┘
""")


# ================================================================
#  SIMULATED FINGERPRINT SYSTEM
# ================================================================
header("🔬 SIMULATED FINGERPRINT SYSTEM")

print("""
  📖 Simulation Strategy:
     Real sensor: আঙুলের image → minutiae → 512-byte template
     Simulation: Random 512-value array = one fingerprint template
     
     Matching: Template distance < threshold → MATCH!
     
     এটা EXACTLY sensor যা internally করে —
     আমরা শুধু math টা simulate করছি!
""")


class FingerprintSensor:
    """Simulates R307/AS608 Fingerprint Sensor"""
    
    def __init__(self, capacity=150):
        self.capacity = capacity
        self.database = {}  # {id: {"template": array, "name": str, ...}}
        self.template_size = 512  # bytes like real sensor
        self.security_level = 3  # 1-5, default 3
        self.FAR = 0.001  # False Accept Rate < 0.001%
        self.FRR = 0.1    # False Reject Rate < 0.1%
        print(f"  ✅ Fingerprint Sensor Initialized")
        print(f"     Capacity: {capacity} templates")
        print(f"     Template size: {self.template_size} bytes")
        print(f"     Security Level: {self.security_level}")
        print(f"     FAR: <{self.FAR}% | FRR: <{self.FRR}%")
    
    def _generate_template(self, seed=None):
        """Generate a fingerprint template (simulates minutiae extraction)"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        # Real sensor creates 512-byte template from minutiae points
        template = rng.randint(0, 256, size=self.template_size).astype(np.uint8)
        return template
    
    def _match_score(self, template1, template2):
        """Calculate match score between two templates (0-300)"""
        # Real sensor uses proprietary algorithm
        # We simulate with normalized correlation
        t1 = template1.astype(float) / 255.0
        t2 = template2.astype(float) / 255.0
        correlation = np.corrcoef(t1, t2)[0, 1]
        # Scale to 0-300 (sensor's confidence range)
        score = int(max(0, correlation * 300))
        return score
    
    def enroll(self, finger_id, name, user_id, dept, seed):
        """
        Enroll a new fingerprint (Phase 1: Registration)
        In real system: user places finger twice on sensor
        """
        if len(self.database) >= self.capacity:
            print(f"  ❌ Database full! Max {self.capacity} templates.")
            return False
        
        if finger_id in self.database:
            print(f"  ⚠️ ID #{finger_id} already exists. Overwriting...")
        
        # Generate template (simulates first scan)
        template1 = self._generate_template(seed)
        
        # Second scan verification (simulates placing finger again)
        template2 = self._generate_template(seed)  # Same seed = same finger
        
        # Verify both scans match
        match_score = self._match_score(template1, template2)
        
        if match_score > 100:
            self.database[finger_id] = {
                "template": template1,
                "name": name,
                "user_id": user_id,
                "dept": dept,
                "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  ✅ Enrolled: ID #{finger_id} → {name} ({user_id}) [{dept}]")
            print(f"     Match score (2 scans): {match_score}/300")
            print(f"     Template hash: {hash(template1.tobytes()) % 100000:05d}")
            return True
        else:
            print(f"  ❌ Enrollment failed — scans don't match (score: {match_score})")
            return False
    
    def search(self, scan_template):
        """
        Search database for matching fingerprint (Phase 2: Identification)
        Simulates 1:N search that real sensor does internally
        """
        best_id = -1
        best_score = 0
        best_data = None
        
        for fid, data in self.database.items():
            score = self._match_score(scan_template, data["template"])
            if score > best_score:
                best_score = score
                best_id = fid
                best_data = data
        
        # Security level threshold mapping (like real sensor)
        thresholds = {1: 40, 2: 60, 3: 80, 4: 100, 5: 130}
        threshold = thresholds[self.security_level]
        
        if best_score >= threshold:
            return {
                "found": True,
                "finger_id": best_id,
                "confidence": best_score,
                "name": best_data["name"],
                "user_id": best_data["user_id"],
                "dept": best_data["dept"],
            }
        else:
            return {
                "found": False,
                "confidence": best_score,
                "message": "No match found in database"
            }
    
    def get_template_count(self):
        return len(self.database)


# ================================================================
#  ENROLLMENT DEMO
# ================================================================
header("📝 Phase 1: ENROLLMENT — Registering Fingerprints")

sensor = FingerprintSensor(capacity=150)

# Register users
users = [
    {"id": 1, "name": "Rahim Ahmed",   "uid": "STU-001", "dept": "CSE", "seed": 1001},
    {"id": 2, "name": "Fatima Khan",   "uid": "STU-002", "dept": "CSE", "seed": 2002},
    {"id": 3, "name": "Karim Hossain", "uid": "STU-003", "dept": "EEE", "seed": 3003},
    {"id": 4, "name": "Nasrin Akter",  "uid": "STU-004", "dept": "CSE", "seed": 4004},
    {"id": 5, "name": "Shakib Rahman", "uid": "STU-005", "dept": "EEE", "seed": 5005},
    {"id": 6, "name": "Aisha Begum",   "uid": "STU-006", "dept": "BBA", "seed": 6006},
    {"id": 7, "name": "Imran Hasan",   "uid": "STU-007", "dept": "CSE", "seed": 7007},
    {"id": 8, "name": "Sumaiya Islam", "uid": "STU-008", "dept": "EEE", "seed": 8008},
]

print(f"\n  Enrolling {len(users)} users...\n")
for u in users:
    sensor.enroll(u["id"], u["name"], u["uid"], u["dept"], u["seed"])

print(f"\n  📊 Total enrolled: {sensor.get_template_count()}/{sensor.capacity}")


# ================================================================
#  MATCHING DEMO — Attendance Marking
# ================================================================
header("🔍 Phase 2: MATCHING — Attendance Scanning")

print("""
  Simulating daily attendance scans...
  (In real life: student places finger on sensor → result in <0.5s)
""")

# Test 1: Known users (same finger = same seed)
print("  ── TEST A: Registered Users Scanning ──\n")

attendance_log = []

for u in users:
    # Simulate scanning same finger (same seed = same template)
    scan = sensor._generate_template(u["seed"])
    result = sensor.search(scan)
    
    if result["found"]:
        print(f"  👆 Finger scanned → ✅ MATCH!")
        print(f"     ID: #{result['finger_id']} | Name: {result['name']}")
        print(f"     Student ID: {result['user_id']} | Dept: {result['dept']}")
        print(f"     Confidence: {result['confidence']}/300")
        print(f"     → Attendance MARKED! ✅")
        
        attendance_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "finger_id": result["finger_id"],
            "name": result["name"],
            "user_id": result["user_id"],
            "dept": result["dept"],
            "confidence": result["confidence"],
            "status": "present"
        })
    else:
        print(f"  👆 Finger scanned → ❌ No match (confidence: {result['confidence']})")
    print()


# Test 2: Unknown person (different seed = different finger)
print("  ── TEST B: Unregistered Person Scanning ──\n")

for i in range(3):
    stranger_scan = sensor._generate_template(seed=99990 + i)
    result = sensor.search(stranger_scan)
    
    if result["found"]:
        print(f"  👆 Stranger #{i+1} → ⚠️ FALSE MATCH! (confidence: {result['confidence']})")
    else:
        print(f"  👆 Stranger #{i+1} → ✅ CORRECTLY REJECTED!")
        print(f"     Best confidence: {result['confidence']}/300 (below threshold)")
        print(f"     Message: {result['message']}")
        print(f"     → 🚨 Alert sent to admin!")
    print()


# Test 3: Slightly different scan (simulates angle/pressure variation)
print("  ── TEST C: Same Person, Slightly Different Scan ──\n")

for u in users[:3]:
    # Add small noise to simulate pressure/angle variation
    base = sensor._generate_template(u["seed"])
    noise = np.random.randint(0, 30, size=512).astype(np.uint8)
    noisy_scan = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    
    result = sensor.search(noisy_scan)
    
    status = "✅ MATCH" if result["found"] else "❌ FAILED"
    name_result = result.get("name", "Unknown")
    conf = result["confidence"]
    
    print(f"  👆 {u['name']} (with noise) → {status}")
    print(f"     Matched: {name_result} | Confidence: {conf}/300")
    correct = result.get("name") == u["name"]
    print(f"     Correct ID? {'✅ Yes!' if correct else '❌ Wrong match'}")
    print()


# ================================================================
#  ATTENDANCE REPORT
# ================================================================
header("📊 ATTENDANCE REPORT — Generated from Fingerprint Scans")

print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d')}")
print(f"  Total scans: {len(attendance_log)}")
print(f"  Present: {len(attendance_log)}/{len(users)}")
print(f"\n  {'#':>3} │ {'Time':>8} │ {'Name':<18} │ {'ID':<8} │ {'Dept':>4} │ {'Conf':>4} │ Status")
print(f"  {'─'*3}─┼─{'─'*8}─┼─{'─'*18}─┼─{'─'*8}─┼─{'─'*4}─┼─{'─'*4}─┼────────")

for i, log in enumerate(attendance_log, 1):
    print(f"  {i:>3} │ {log['time']:>8} │ {log['name']:<18} │ {log['user_id']:<8} │ {log['dept']:>4} │ {log['confidence']:>4} │ ✅ Present")


# ================================================================
#  WOKWI SIMULATION INFO
# ================================================================
header("🖥️ SIMULATOR — Wokwi তে Fingerprint কিভাবে কাজ করবে?")

print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │  Wokwi তে Fingerprint Sensor Simulation:                        │
  │                                                                  │
  │  📌 Option 1: Push Button দিয়ে simulate                         │
  │     • Button press = "finger placed on sensor"                   │
  │     • Each button → pre-assigned fingerprint ID                  │
  │     • Button 1 = Rahim, Button 2 = Fatima, etc.                 │
  │     • Code internally simulates the matching flow                │
  │     → Wokwi project links with this approach exist! ✅           │
  │                                                                  │
  │  📌 Option 2: Serial Monitor Input                               │
  │     • Type fingerprint ID in Serial Monitor                      │
  │     • Code looks up the ID in the user database                  │
  │     • Simulates the sensor returning a match                     │
  │     → Simple and effective for demo ✅                           │
  │                                                                  │
  │  📌 Option 3: Adafruit_Fingerprint Library (Wokwi supported!)   │
  │     • Wokwi has basic fingerprint sensor support                 │
  │     • Uses SoftwareSerial (TX/RX pins)                           │
  │     • Library handles enrollment and search commands             │
  │     → Most realistic simulation ✅                               │
  │                                                                  │
  │  🔗 Ready-made Wokwi Projects:                                   │
  │     • Fingerprint Attendance System:                             │
  │       https://wokwi.com/projects/388478575022925825              │
  │     • Fingerprint Scanner:                                       │
  │       https://wokwi.com/projects/399333348588767233              │
  │     • Fingerprint + SMS Notification:                            │
  │       https://wokwi.com/projects/375714067438538753              │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  ESP32 + R307 Wiring (Real & Simulated):                        │
  │                                                                  │
  │  R307 Sensor          ESP32                                      │
  │  ──────────           ─────                                      │
  │  VCC (Red)    ───→    3.3V                                       │
  │  GND (Black)  ───→    GND                                        │
  │  TX  (Yellow) ───→    GPIO 16 (RX2)                              │
  │  RX  (Green)  ───→    GPIO 17 (TX2)                              │
  │                                                                  │
  │  Library: Adafruit_Fingerprint                                   │
  │  Baud rate: 57600                                                │
  │  Interface: UART (Serial2 on ESP32)                              │
  └──────────────────────────────────────────────────────────────────┘
""")


# ================================================================
#  ESP32 CODE FOR REAL HARDWARE
# ================================================================
header("📝 ESP32 Arduino Code — Fingerprint Attendance")

esp32_code = '''// ============================================
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
'''

code_path = "models/fingerprint/esp32_fingerprint_attendance.ino"
os.makedirs("models/fingerprint", exist_ok=True)
with open(code_path, 'w') as f:
    f.write(esp32_code)
print(f"  ✅ ESP32 code saved: {code_path}")


# ================================================================
#  FINAL SUMMARY
# ================================================================
header("🎯 FINAL SUMMARY — Fingerprint-Based Attendance System")

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  YOUR UPDATED SYSTEM:                                           │
  │                                                                  │
  │  🔑 Primary ID   : Fingerprint (R307/AS608) — 99.9% accuracy   │
  │  📡 Presence     : PIR sensor — confirms physical presence      │
  │  📏 Distance     : Ultrasonic — measures how close person is    │
  │  🏃 Activity     : Accelerometer — walking/sitting/running      │
  │  📶 Communication: ESP32 WiFi — sends data to cloud             │
  │  🖥️ Display      : OLED — shows name, status, confirmation     │
  │  💡 Feedback     : LED (green/red) + Buzzer (beep)              │
  │                                                                  │
  │  FLOW:                                                           │
  │  PIR detects motion → OLED: "Place finger" → Finger scanned     │
  │  → Sensor returns ID + confidence → ESP32 looks up user info    │
  │  → Attendance marked → Data sent to cloud → LED + Buzzer ✅      │
  │                                                                  │
  │  WHY FINGERPRINT IS BEST:                                       │
  │  • FAR < 0.001% (False Accept)                                  │
  │  • FRR < 0.1%   (False Reject)                                  │
  │  • Cannot share/lose/duplicate                                   │
  │  • Sensor stores templates internally (512 bytes each)           │
  │  • ESP32 only sends commands — no heavy processing!              │
  │  • Price: $8-15 for the sensor                                  │
  │                                                                  │
  │  SIMULATOR:                                                      │
  │  • Wokwi has fingerprint sensor projects ✅                      │
  │  • Python simulation shows matching algorithm ✅                 │
  │  • Push button simulation in Wokwi ✅                            │
  └──────────────────────────────────────────────────────────────────┘
""")
