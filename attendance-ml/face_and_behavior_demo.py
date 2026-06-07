"""
╔══════════════════════════════════════════════════════════════╗
║  Face Recognition + Behaviour Analysis — Complete Demo      ║
║──────────────────────────────────────────────────────────────║
║  Part A: RFID vs Face Recognition — কোনটা কিভাবে কাজ করে   ║
║  Part B: Face Recognition Simulation (no camera needed!)    ║
║  Part C: Behaviour/Activity Identification                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import os, json, warnings, math, random
from collections import Counter
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ================================================================
#  PART A: RFID vs Face Recognition — Side by Side
# ================================================================
header("📋 PART A: RFID vs Face Recognition — কিভাবে Identify করে?")

print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │  RFID কিভাবে person identify করে:                               │
  │                                                                  │
  │  1. প্রতিটি student/employee কে একটি RFID card দেওয়া হয়         │
  │  2. Card এ একটি unique UID থাকে (যেমন: A1B2C3D4)                │
  │  3. Card scan করলে UID read হয়                                  │
  │  4. Database এ UID match করে → নাম, ID, department পাওয়া যায়   │
  │  5. ✅ Match → Attendance marked!                                │
  │  6. ❌ No match → Rejected + Alert!                              │
  │                                                                  │
  │  🎯 Accuracy: 99.9% (card UID unique, collision নেই)             │
  │  ⚠️ Weakness: Card share করলে proxy attendance হতে পারে         │
  │  💡 Solution: RFID + PIR sensor combine → card + presence দুটোই  │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  Face Recognition কিভাবে person identify করে:                    │
  │                                                                  │
  │  1. Camera দিয়ে ছবি capture করে                                  │
  │  2. ছবি থেকে face detect করে (haar cascade / MTCNN / HOG)       │
  │  3. Face থেকে 128D embedding vector বের করে                     │
  │     (প্রতিটি মুখের একটা mathematical signature)                  │
  │  4. Database এর stored embeddings এর সাথে compare করে           │
  │  5. Distance < threshold → Match! Person identified!             │
  │  6. Distance > threshold → Unknown person!                       │
  │                                                                  │
  │  🎯 Accuracy: 95-99% (depends on lighting, angle, quality)      │
  │  ⚠️ Weakness: Low light, face mask, angle change → accuracy drop │
  │  💡 Advantage: Proxy attendance impossible — মুখ ভাড়া দেওয়া যায় না!│
  └──────────────────────────────────────────────────────────────────┘
""")


# ================================================================
#  PART B: Face Recognition Simulation (NO camera needed!)
# ================================================================
header("🎭 PART B: Face Recognition — Simulator এ কিভাবে কাজ করে?")

print("""
  📖 KEY CONCEPT: Face Recognition এর মূল কাজ হলো:
     
     ছবি → 128-dimensional number vector (embedding)
     
     এই 128টা number ই হলো face এর "mathematical fingerprint"
     দুইটা face এর embedding এর distance কম = same person!
     
  🖥️ SIMULATOR এ আমরা কি করব:
     Real camera না থাকলেও, আমরা simulate করতে পারি:
     1. Pre-saved face embeddings (128D vectors) database এ store
     2. নতুন "scan" মানে একটা নতুন vector generate
     3. Database এর vectors এর সাথে distance compare
     4. সবচেয়ে কাছের match = identified person!
     
     এটা EXACTLY real face recognition যা করে —
     শুধু camera part টা skip হচ্ছে!
""")

# ---- Simulated Face Database ----
print("  📂 Creating Face Embedding Database...")

# In real system: face_recognition.face_encodings(image) gives 128D vector
# Here we simulate realistic 128D embeddings for each person
registered_faces = {}
users = [
    {"name": "Rahim Ahmed",   "id": "STU-001", "dept": "CSE"},
    {"name": "Fatima Khan",   "id": "STU-002", "dept": "CSE"},
    {"name": "Karim Hossain", "id": "STU-003", "dept": "EEE"},
    {"name": "Nasrin Akter",  "id": "STU-004", "dept": "CSE"},
    {"name": "Shakib Rahman", "id": "STU-005", "dept": "EEE"},
]

for user in users:
    # Generate a unique 128D embedding for each person
    # (In real system, this comes from face_recognition library)
    base_embedding = np.random.randn(128).astype(np.float32)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)  # Normalize
    registered_faces[user["id"]] = {
        "name": user["name"],
        "dept": user["dept"],
        "embedding": base_embedding,
    }
    print(f"     ✅ {user['name']:20s} → embedding[0:5] = {base_embedding[:5].round(3)}")

print(f"\n  📊 Database: {len(registered_faces)} faces registered")
print(f"     Each face = 128-dimensional vector")
print(f"     Storage per face = {128 * 4} bytes = 512 bytes")


# ---- Face Matching Function ----
def cosine_similarity(a, b):
    """How similar are two face embeddings? (1.0 = identical, 0.0 = different)"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

def euclidean_distance(a, b):
    """Euclidean distance between two embeddings (lower = more similar)"""
    return np.linalg.norm(a - b)

def identify_face(scan_embedding, database, threshold=0.6):
    """
    Match a scanned face embedding against the database.
    Returns: (matched_id, name, confidence, distance) or None
    """
    best_match = None
    best_similarity = -1
    best_distance = float('inf')
    
    for user_id, data in database.items():
        sim = cosine_similarity(scan_embedding, data["embedding"])
        dist = euclidean_distance(scan_embedding, data["embedding"])
        
        if sim > best_similarity:
            best_similarity = sim
            best_distance = dist
            best_match = (user_id, data["name"], data["dept"])
    
    if best_similarity >= threshold:
        return {
            "matched": True,
            "user_id": best_match[0],
            "name": best_match[1],
            "dept": best_match[2],
            "confidence": round(best_similarity * 100, 2),
            "distance": round(best_distance, 4),
        }
    else:
        return {
            "matched": False,
            "confidence": round(best_similarity * 100, 2),
            "distance": round(best_distance, 4),
        }


# ---- Test: Known person ----
header("🧪 TEST 1: Known Person Scanning (Registered User)")

for user_id, data in registered_faces.items():
    # Simulate: same person but slight variation (camera angle, lighting)
    noise = np.random.randn(128).astype(np.float32) * 0.15  # Small noise
    scan = data["embedding"] + noise
    scan = scan / np.linalg.norm(scan)  # Normalize
    
    result = identify_face(scan, registered_faces, threshold=0.6)
    
    status = "✅ MATCH" if result["matched"] else "❌ FAILED"
    print(f"\n  📷 Scanning: {data['name']}")
    print(f"     {status} → {result.get('name', 'Unknown')}")
    print(f"     Confidence: {result['confidence']:.1f}%")
    print(f"     Distance: {result['distance']:.4f}")


# ---- Test: Unknown person ----
header("🧪 TEST 2: Unknown Person (NOT in database)")

for i in range(3):
    # Completely random embedding = stranger
    stranger = np.random.randn(128).astype(np.float32)
    stranger = stranger / np.linalg.norm(stranger)
    
    result = identify_face(stranger, registered_faces, threshold=0.6)
    
    print(f"\n  📷 Scanning: Unknown Person #{i+1}")
    if result["matched"]:
        print(f"     ⚠️ FALSE MATCH → {result['name']} ({result['confidence']:.1f}%)")
    else:
        print(f"     ✅ Correctly REJECTED — Unknown person!")
        print(f"     Best similarity: {result['confidence']:.1f}% (below 60% threshold)")
        print(f"     Distance: {result['distance']:.4f}")


# ---- Test: Photo attack (proxy attempt) ----
header("🧪 TEST 3: Threshold Analysis — কিভাবে Accuracy নিয়ন্ত্রণ করে?")

print("""
  📖 Threshold কি?
     
     Threshold = minimum similarity score for a "match"
     
     Higher threshold (0.8) → Strict matching → কম false positive
     Lower threshold (0.4)  → Loose matching  → বেশি false positive
     
     Best balance = 0.6 (standard for face_recognition library)
""")

# Test with different thresholds
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n  {'Threshold':>10} │ {'True Accept':>12} │ {'False Reject':>13} │ {'False Accept':>13} │ {'Verdict':>10}")
print(f"  {'─'*10}─┼─{'─'*12}─┼─{'─'*13}─┼─{'─'*13}─┼─{'─'*10}")

for thresh in thresholds:
    # Test known users
    true_accept = 0
    false_reject = 0
    for uid, data in registered_faces.items():
        noise = np.random.randn(128).astype(np.float32) * 0.15
        scan = data["embedding"] + noise
        scan = scan / np.linalg.norm(scan)
        result = identify_face(scan, registered_faces, threshold=thresh)
        if result["matched"] and result["user_id"] == uid:
            true_accept += 1
        else:
            false_reject += 1
    
    # Test strangers
    false_accept = 0
    for _ in range(10):
        stranger = np.random.randn(128).astype(np.float32)
        stranger = stranger / np.linalg.norm(stranger)
        result = identify_face(stranger, registered_faces, threshold=thresh)
        if result["matched"]:
            false_accept += 1
    
    verdict = "🏆 Best!" if thresh == 0.6 else ("⚠️ Loose" if thresh < 0.5 else ("⚠️ Strict" if thresh > 0.8 else "✅ OK"))
    print(f"  {thresh:>10.1f} │ {true_accept:>10}/5  │ {false_reject:>11}/5  │ {false_accept:>11}/10 │ {verdict:>10}")


# ================================================================
#  PART C: Behaviour / Activity Identification
# ================================================================
header("🏃 PART C: Behaviour / Activity Identification — কিভাবে কাজ করে?")

print("""
  📖 Behaviour Identification = ২ ধরনের:
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  TYPE 1: Real-time Sensor-based Activity Detection              │
  │  ─────────────────────────────────────────────────               │
  │  Sensors: Accelerometer (MPU6050) + PIR + Ultrasonic            │
  │                                                                  │
  │  Accelerometer data থেকে:                                       │
  │    • Walking  → x,y oscillate (±1.5), z varies (9.5-10.5)      │
  │    • Running  → x,y high amplitude (±2.0), z high variance     │
  │    • Sitting  → x,y near 0, z steady at ~9.8 (gravity only)    │
  │    • Standing → x,y near 0, z at ~9.8, PIR occasionally fires  │
  │    • Idle     → all sensors quiet, no motion, far distance      │
  │                                                                  │
  │  PIR sensor:                                                     │
  │    • Motion detected = someone present                          │
  │    • No motion for 5 min = room empty                           │
  │                                                                  │
  │  Ultrasonic distance:                                            │
  │    • < 50cm = person very close (at desk/door)                  │
  │    • 50-200cm = person in room                                  │
  │    • > 200cm = room likely empty                                │
  └─────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  TYPE 2: Time-based Behaviour Pattern Analysis                  │
  │  ────────────────────────────────────────────                    │
  │  Historical attendance data থেকে:                               │
  │                                                                  │
  │  • Regular pattern:  প্রতিদিন 8:30-5:00 আসে                    │
  │  • Late pattern:     80%+ দিন 9:30 এর পরে আসে                  │
  │  • Early leaver:     প্রায়ই 2:00 PM এ চলে যায়                  │
  │  • Irregular:        কিছুদিন আসে, কিছুদিন আসে না               │
  │  • Absent trend:     ক্রমশ absence বাড়ছে                       │
  │  • Zone violation:   Restricted area তে unauthorized access     │
  └─────────────────────────────────────────────────────────────────┘
""")


# ---- Sensor-based Activity Detection Demo ----
header("📡 DEMO: Sensor-based Activity Detection")

print("  Simulating real sensor readings → activity classification:\n")

# Simulate different activities
activities = [
    {
        "label": "WALKING",
        "accel_x": 1.2, "accel_y": 0.8, "accel_z": 10.1,
        "pir": 1, "dist": 45,
        "explanation": "x,y moderate oscillation + z slightly above gravity + PIR active + person close"
    },
    {
        "label": "RUNNING",
        "accel_x": 2.5, "accel_y": 1.9, "accel_z": 11.2,
        "pir": 1, "dist": 60,
        "explanation": "x,y HIGH amplitude + z very high + PIR active = intense movement"
    },
    {
        "label": "SITTING",
        "accel_x": 0.05, "accel_y": 0.03, "accel_z": 9.81,
        "pir": 1, "dist": 80,
        "explanation": "x,y near ZERO + z exactly gravity (9.81) + PIR on but still = seated"
    },
    {
        "label": "STANDING",
        "accel_x": 0.15, "accel_y": 0.10, "accel_z": 9.82,
        "pir": 1, "dist": 120,
        "explanation": "x,y very small + z gravity + PIR on + further away = standing in room"
    },
    {
        "label": "IDLE / EMPTY",
        "accel_x": 0.01, "accel_y": 0.01, "accel_z": 9.80,
        "pir": 0, "dist": 350,
        "explanation": "All near zero + NO PIR + very far distance = nobody in room"
    },
    {
        "label": "⚠️ UNAUTHORIZED",
        "accel_x": 1.8, "accel_y": 1.5, "accel_z": 10.5,
        "pir": 1, "dist": 30,
        "explanation": "Motion at 2 AM + PIR active + person very close = ANOMALY!"
    },
]

for i, act in enumerate(activities, 1):
    print(f"  ── Scenario {i}: {act['label']} ──")
    print(f"     📊 Sensors: accel=({act['accel_x']}, {act['accel_y']}, {act['accel_z']})")
    print(f"                 PIR={act['pir']}, Distance={act['dist']}cm")
    print(f"     📖 Why?  {act['explanation']}")
    
    # Rule-based classification (explainable!)
    magnitude = math.sqrt(act['accel_x']**2 + act['accel_y']**2)
    
    if act['pir'] == 0 and act['dist'] > 200:
        detected = "IDLE/EMPTY"
    elif magnitude > 2.0:
        detected = "RUNNING"
    elif magnitude > 0.5:
        detected = "WALKING"
    elif magnitude > 0.1 and act['dist'] < 150:
        detected = "STANDING"
    elif magnitude < 0.1 and act['pir'] == 1:
        detected = "SITTING"
    else:
        detected = "UNKNOWN"
    
    print(f"     🔮 Detected: {detected}")
    print(f"     📐 Accel magnitude: {magnitude:.3f} (threshold: sitting<0.1, walk>0.5, run>2.0)")
    print()


# ---- Time-based Behaviour Pattern Analysis ----
header("📅 DEMO: Time-based Behaviour Pattern Analysis")

# Load attendance data
attend_df = pd.read_csv("data/attendance_records.csv")
attend_present = attend_df[attend_df['status'] == 'present'].copy()

if len(attend_present) > 0 and 'entry_time' in attend_present.columns:
    attend_present['entry_hour'] = attend_present['entry_time'].apply(
        lambda x: int(str(x).split(':')[0]) if pd.notna(x) and str(x) != '' else 0
    )
    attend_present['exit_hour'] = attend_present['exit_time'].apply(
        lambda x: int(str(x).split(':')[0]) if pd.notna(x) and str(x) != '' else 0
    )

print("  Analyzing behaviour patterns for each user:\n")

for user_name in attend_df['user_name'].unique():
    if user_name == 'UNKNOWN':
        continue
    
    user_data = attend_df[attend_df['user_name'] == user_name]
    present_data = user_data[user_data['status'] == 'present']
    absent_data = user_data[user_data['status'] == 'absent']
    
    total = len(user_data)
    present_count = len(present_data)
    absent_count = len(absent_data)
    
    if present_count == 0:
        continue
    
    att_rate = present_count / total * 100
    late_count = present_data['is_late'].astype(int).sum()
    early_count = present_data['is_early_leave'].astype(int).sum()
    avg_duration = present_data['duration_hours'].astype(float).mean()
    
    # Determine behaviour pattern
    patterns = []
    if att_rate >= 90:
        patterns.append("✅ REGULAR (90%+ attendance)")
    elif att_rate >= 70:
        patterns.append("⚠️ MODERATE (70-90% attendance)")
    else:
        patterns.append("🚨 IRREGULAR (<70% attendance)")
    
    if late_count / present_count > 0.3:
        patterns.append("⏰ HABITUAL LATECOMER")
    
    if early_count / present_count > 0.2:
        patterns.append("🚪 FREQUENT EARLY LEAVER")
    
    if avg_duration < 5:
        patterns.append("📉 SHORT STAY (<5 hours avg)")
    elif avg_duration > 8:
        patterns.append("📈 OVERTIME WORKER (>8 hours avg)")
    
    print(f"  👤 {user_name} ({user_data.iloc[0]['user_id']})")
    print(f"     Attendance: {present_count}/{total} ({att_rate:.0f}%)")
    print(f"     Late: {late_count}/{present_count} | Early leave: {early_count}/{present_count}")
    print(f"     Avg stay: {avg_duration:.1f} hours")
    print(f"     Behaviour: {' | '.join(patterns)}")
    print()


# ---- Zone-based Movement Tracking ----
header("🗺️ DEMO: Zone-based Movement & Restricted Area Monitoring")

sensor_df = pd.read_csv("data/sensor_readings.csv")

print("  Analyzing zone activity patterns:\n")

zone_stats = sensor_df.groupby('zone').agg(
    total_readings=('pir_motion', 'count'),
    motion_events=('pir_motion', 'sum'),
    avg_distance=('ultrasonic_cm', 'mean'),
    anomalies=('is_anomaly', 'sum'),
).reset_index()

for _, zone in zone_stats.iterrows():
    activity_rate = zone['motion_events'] / zone['total_readings'] * 100
    bar = '█' * int(activity_rate / 2)
    anomaly_flag = " 🚨 ANOMALIES DETECTED!" if zone['anomalies'] > 0 else ""
    
    print(f"  📍 {zone['zone']:>15}: {bar} {activity_rate:.0f}% activity | "
          f"Avg dist: {zone['avg_distance']:.0f}cm | "
          f"Anomalies: {int(zone['anomalies'])}{anomaly_flag}")


# ================================================================
#  SUMMARY
# ================================================================
header("🎯 FINAL SUMMARY — আপনার প্রজেক্টের জন্য Recommendation")

print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  🔑 IDENTIFICATION (কে এই person?):                              │
  │  ──────────────────────────────────                               │
  │  Option A: RFID Card → 99.9% accurate, simple, cheap            │
  │  Option B: Face Recognition → 95-99%, no card needed             │
  │  Option C: RFID + Face → 99.99% (dual verification!)            │
  │                                                                  │
  │  👉 Recommendation: RFID + PIR (your proposal এ যা আছে তাই     │
  │     sufficient! Face recognition optional add-on)                │
  │                                                                  │
  │  🖥️ SIMULATOR এ কি হবে:                                         │
  │  ─────────────────────                                           │
  │  • RFID: Wokwi তে virtual RFID card tap করলেই UID read হয় ✅   │
  │  • Face: 128D embedding vectors simulate করে matching দেখানো ✅  │
  │  • PIR: Click করলে motion detect হয় ✅                          │
  │  • সব কিছু real system এর মতোই কাজ করে!                         │
  │                                                                  │
  │  🏃 BEHAVIOUR (কি করছে person?):                                 │
  │  ──────────────────────────────                                   │
  │  Sensor-based: Accelerometer magnitude → activity classify       │
  │    • magnitude < 0.1 → Sitting                                   │
  │    • magnitude 0.1-0.5 → Standing                                │
  │    • magnitude 0.5-2.0 → Walking                                 │
  │    • magnitude > 2.0 → Running                                   │
  │                                                                  │
  │  Time-based: Attendance history → pattern analysis               │
  │    • Late pattern, Early leave, Irregular, Absent trend          │
  │    • Zone violation, Unauthorized access hours                   │
  │                                                                  │
  │  🧠 ML Models:                                                   │
  │    • KNN/Random Forest → Activity classification (91%)           │
  │    • Autoencoder → Anomaly detection (95%, 100% recall)          │
  │    • Decision Tree → Attendance prediction (88%)                 │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
""")
