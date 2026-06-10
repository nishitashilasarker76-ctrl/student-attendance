#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

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

const char* FIREBASE_URL = "https://smart-attendance-and-activity-default-rtdb.asia-southeast1.firebasedatabase.app/";

// ===== EMPLOYEE DATABASE =====
// Blue card(b)=Rahim, Green(g)=Fatima, Yellow(y)=Karim, Red(r)=Unknown
struct Employee {
  const char* uid;
  const char* name;
  const char* id;
  const char* dept;
  bool attended;
};

Employee employees[] = {
  {"", "Rahim Ahmed",   "EMP-001", "IT Dept",  false},
  {"", "Fatima Khan",   "EMP-002", "HR Dept",  false},
  {"", "Karim Hossain", "EMP-003", "Finance",  false},
};
int NUM_EMP = 3;
int regCount = 0;
int attCount = 0;
bool waitCard = false;
unsigned long mTime = 0;

// ===== AUTO ACTIVITY (UCI HAR 6 classes) =====
const char* acts[] = {"WALKING","WALK_UP","WALK_DOWN","SITTING","STANDING","LAYING"};
float accD[][3] = {{0.28,-0.02,-0.07},{0.26,-0.03,-0.12},{0.31,-0.02,-0.05},{0.27,-0.01,-0.10},{0.28,-0.02,-0.09},{0.28,-0.03,-0.02}};
int curAct = 0;
unsigned long lastAct = 0;

const char* getAct() {
  if (millis()-lastAct > random(2000,6000)) { curAct=random(0,6); lastAct=millis(); }
  return acts[curAct];
}
void getAcc(float &x,float &y,float &z) {
  x=accD[curAct][0]+random(-50,50)/1000.0;
  y=accD[curAct][1]+random(-50,50)/1000.0;
  z=accD[curAct][2]+random(-50,50)/1000.0;
}

float getDist() {
  digitalWrite(TRIG_PIN,LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN,HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN,LOW);
  return pulseIn(ECHO_PIN,HIGH,30000)*0.034/2.0;
}

String getUID() {
  String u="";
  for(byte i=0;i<rfid.uid.size;i++){
    if(rfid.uid.uidByte[i]<0x10) u+="0";
    u+=String(rfid.uid.uidByte[i],HEX);
  }
  u.toUpperCase();
  return u;
}

int findEmp(String u) {
  for(int i=0;i<NUM_EMP;i++) if(String(employees[i].uid)==u) return i;
  return -1;
}

int autoReg(String u) {
  if(regCount>=NUM_EMP) return -1;
  employees[regCount].uid=strdup(u.c_str());
  Serial.print("  Registered: "); Serial.print(u);
  Serial.print(" -> "); Serial.println(employees[regCount].name);
  regCount++;
  return regCount-1;
}

// ===== FIREBASE =====
void sendFB(Employee* e,float d,const char* a) {
  if(WiFi.status()!=WL_CONNECTED) return;
  HTTPClient h;
  String url=String(FIREBASE_URL)+"/attendance/"+String(e->id)+".json";
  float ax,ay,az; getAcc(ax,ay,az);
  String j="{\"name\":\""+String(e->name)+"\",\"employee_id\":\""+String(e->id)+"\",\"department\":\""+String(e->dept)+"\",\"rfid_uid\":\""+String(e->uid)+"\",\"status\":\"present\",\"distance_cm\":"+String((int)d)+",\"activity\":\""+String(a)+"\",\"accel_x\":"+String(ax,3)+",\"accel_y\":"+String(ay,3)+",\"accel_z\":"+String(az,3)+",\"attendance\":"+String(attCount)+"}";
  h.begin(url); h.addHeader("Content-Type","application/json");
  int c=h.PUT(j);
  Serial.println(c==200?"  Firebase: SAVED!":"  Firebase: Error");
  h.end();
}

void alertFB(String u,float d,const char* a) {
  if(WiFi.status()!=WL_CONNECTED) return;
  HTTPClient h;
  String url=String(FIREBASE_URL)+"/alerts/latest.json";
  String j="{\"type\":\"UNAUTHORIZED\",\"uid\":\""+u+"\",\"dist\":"+String((int)d)+",\"act\":\""+String(a)+"\",\"msg\":\"Unknown card!\"}";
  h.begin(url); h.addHeader("Content-Type","application/json"); h.PUT(j); h.end();
}

// ===== DISPLAYS =====
void showLive() {
  float d=getDist(); int p=digitalRead(PIR_PIN);
  const char* a=getAct(); float ax,ay,az; getAcc(ax,ay,az);
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0);  display.println("= OFFICE MONITORING =");
  display.drawLine(0,10,128,10,WHITE);
  display.setCursor(0,13); display.print("PIR: "); display.println(p?"MOTION!":"No motion");
  display.setCursor(0,23); display.print("Dist: "); display.print((int)d); display.println(" cm");
  display.setCursor(0,33); display.print("Acc:"); display.print(ax,2); display.print(","); display.print(ay,2); display.print(","); display.println(az,2);
  display.setCursor(0,43); display.print(">> "); display.print(a); display.println(" <<");
  display.setCursor(0,55); display.print(attCount); display.print("/3|Emp:"); display.print(regCount); display.print("|WiFi");
  display.display();
}

void showTap() {
  float d=getDist(); const char* a=getAct();
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0); display.println("!! EMPLOYEE DETECTED !!");
  display.drawLine(0,10,128,10,WHITE);
  display.setCursor(0,14); display.print("Dist:"); display.print((int)d); display.print("cm "); display.println(a);
  display.drawLine(0,28,128,28,WHITE);
  display.setTextSize(1);
  display.setCursor(0,32); display.println("TAP YOUR ID CARD");
  display.setCursor(0,44); display.println("Keyboard shortcuts:");
  display.setCursor(0,54); display.println("b=Blue g=Green y=Yellow");
  display.display();
}

void showWelcome(Employee* e,float d,const char* a) {
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0); display.println("== WELCOME EMPLOYEE ==");
  display.setTextSize(2);
  display.setCursor(0,11); display.println(e->name);
  display.setTextSize(1);
  display.setCursor(0,32); display.print(e->id); display.print("|"); display.println(e->dept);
  display.setCursor(0,42); display.print("UID:"); display.println(e->uid);
  display.setCursor(0,52); display.print("Act:"); display.print(a); display.print(" #"); display.print(attCount); display.print("/3");
  display.display();
}

void showAlready(Employee* e) {
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(0,0); display.println("!! ALREADY CHECKED IN !!");
  display.setTextSize(2);
  display.setCursor(0,14); display.println(e->name);
  display.setTextSize(1);
  display.setCursor(0,38); display.print(e->id); display.print("|"); display.println(e->dept);
  display.setCursor(0,50); display.println("Already recorded today!");
  display.display();
}

void showDenied(String u) {
  display.clearDisplay();
  display.setTextSize(2); display.setTextColor(WHITE);
  display.setCursor(10,3); display.println("ACCESS");
  display.setCursor(10,23); display.println("DENIED!");
  display.setTextSize(1);
  display.setCursor(0,45); display.print("UID:"); display.println(u);
  display.setCursor(0,55); display.println("Not registered! ALERT!");
  display.display();
}

void bOK(){tone(BUZZER_PIN,1000,150);delay(200);tone(BUZZER_PIN,1500,150);delay(200);noTone(BUZZER_PIN);}
void bDup(){tone(BUZZER_PIN,800,100);delay(150);tone(BUZZER_PIN,800,100);delay(150);noTone(BUZZER_PIN);}
void bNo(){tone(BUZZER_PIN,300,400);delay(500);noTone(BUZZER_PIN);}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(0));
  Serial.println("\n========================================");
  Serial.println("  EMPLOYEE ATTENDANCE & ACTIVITY");
  Serial.println("  TRACKING SYSTEM (RFID)");
  Serial.println("========================================");
  Serial.println("  HOW TO TAP CARDS:");
  Serial.println("    Press 'b' = Blue card (Rahim)");
  Serial.println("    Press 'g' = Green card (Fatima)");
  Serial.println("    Press 'y' = Yellow card (Karim)");
  Serial.println("    Press 'r' = Red card (UNKNOWN!)");
  Serial.println("    Press 't' = Quick tap");
  Serial.println("========================================\n");

  pinMode(PIR_PIN,INPUT);
  pinMode(TRIG_PIN,OUTPUT); pinMode(ECHO_PIN,INPUT);
  pinMode(LED_GREEN,OUTPUT); pinMode(LED_RED,OUTPUT);
  pinMode(BUZZER_PIN,OUTPUT);

  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextSize(1); display.setTextColor(WHITE);
  display.setCursor(5,3);  display.println("EMPLOYEE ATTENDANCE");
  display.setCursor(5,15); display.println("& Activity Tracking");
  display.drawLine(0,25,128,25,WHITE);
  display.setCursor(5,29); display.println("RFID Authentication");
  display.setCursor(5,41); display.println("b=Rahim g=Fatima");
  display.setCursor(5,51); display.println("y=Karim r=Unknown");
  display.display();
  delay(3000);

  SPI.begin();
  rfid.PCD_Init();
  Serial.println("RFID: Ready");

  display.clearDisplay();
  display.setCursor(5,15); display.println("Connecting WiFi...");
  display.display();
  WiFi.begin("Wokwi-GUEST","");
  while(WiFi.status()!=WL_CONNECTED) delay(500);
  Serial.println("WiFi: Connected!\n");

  display.setCursor(5,30); display.println("WiFi: Connected!");
  display.setCursor(5,45); display.println("System Ready!");
  display.display();
  delay(1500);
}

// ===== MAIN LOOP =====
void loop() {
  int motion=digitalRead(PIR_PIN);

  if(!waitCard) { showLive(); delay(400); }

  if(motion==HIGH && !waitCard) {
    showTap();
    waitCard=true;
    mTime=millis();
    Serial.println("----------------------------------");
    Serial.println("Employee detected!");
    Serial.println("Press b/g/y/r on keyboard to tap card");
  }

  if(waitCard) {
    if(millis()-mTime>15000) { waitCard=false; return; }

    if(!rfid.PICC_IsNewCardPresent()) return;
    if(!rfid.PICC_ReadCardSerial()) return;

    String uid=getUID();
    float dist=getDist();
    const char* act=getAct();

    Serial.print("Card tapped! UID: "); Serial.println(uid);

    int idx=findEmp(uid);
    if(idx==-1) idx=autoReg(uid);

    if(idx>=0) {
      if(employees[idx].attended) {
        Serial.print(employees[idx].name); Serial.println(" -> ALREADY!");
        digitalWrite(LED_RED,HIGH);
        showAlready(&employees[idx]);
        bDup();
        delay(3000);
        digitalWrite(LED_RED,LOW);
      } else {
        attCount++;
        employees[idx].attended=true;
        Serial.print("CHECK IN: "); Serial.print(employees[idx].name);
        Serial.print(" ("); Serial.print(employees[idx].dept);
        Serial.print(") Act:"); Serial.println(act);
        digitalWrite(LED_GREEN,HIGH);
        showWelcome(&employees[idx],dist,act);
        bOK();
        sendFB(&employees[idx],dist,act);
        delay(3000);
        digitalWrite(LED_GREEN,LOW);
      }
    } else {
      Serial.print("REJECTED! UID: "); Serial.println(uid);
      digitalWrite(LED_RED,HIGH);
      showDenied(uid);
      bNo();
      alertFB(uid,dist,act);
      delay(3000);
      digitalWrite(LED_RED,LOW);
    }

    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
    waitCard=false;
    Serial.println("----------------------------------\n");
  }
  delay(100);
}