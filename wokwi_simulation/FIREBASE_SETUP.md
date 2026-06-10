# 🔥 Firebase Setup — Step by Step (10 minutes)

## Step 1: Firebase Project তৈরি
1. Go to: https://console.firebase.google.com
2. Click "Add project"
3. Name: `attendance-system`
4. Disable Google Analytics (not needed)
5. Click "Create project"

## Step 2: Realtime Database Enable
1. Left sidebar → "Build" → "Realtime Database"
2. Click "Create Database"
3. Location: Choose nearest (Singapore for Bangladesh)
4. Security rules → "Start in TEST mode" (for development)
5. Click "Enable"

## Step 3: Get Your Database URL
After creating, you'll see something like:
```
https://attendance-system-12345-default-rtdb.firebaseio.com/
```
☝️ THIS is your FIREBASE_HOST — copy this!

## Step 4: Update Arduino Code
In `sketch.ino`, replace this line:
```cpp
const char* FIREBASE_HOST = "https://YOUR-PROJECT-ID.firebaseio.com";
```
With YOUR actual URL:
```cpp
const char* FIREBASE_HOST = "https://attendance-system-12345-default-rtdb.firebaseio.com";
```

## Step 5: Test
Run Wokwi simulation → press a button → check Firebase console → data should appear!

## 📊 Firebase Database Structure (auto-created):
```
attendance-system-12345/
├── attendance/
│   ├── STU-001/
│   │   ├── student_id: "STU-001"
│   │   ├── name: "Rahim Ahmed"
│   │   ├── department: "CSE"
│   │   ├── date: "2026-06-08"
│   │   ├── time: "08:45:23"
│   │   ├── status: "present"
│   │   ├── confidence: 285
│   │   ├── distance_cm: 42.5
│   │   └── pir_motion: true
│   ├── STU-002/
│   │   └── ...
│   └── STU-003/
│       └── ...
└── alerts/
    └── latest/
        ├── type: "UNAUTHORIZED_ACCESS"
        ├── message: "Unknown fingerprint detected!"
        └── timestamp: 123456789
```

## ⚠️ Important Notes:
- TEST mode rules expire after 30 days — enough for thesis!
- Free tier: 1GB storage, 10GB/month download — more than enough!
- Wokwi's "Wokwi-GUEST" WiFi connects to REAL internet → Firebase works!
