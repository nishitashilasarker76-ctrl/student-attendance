"""
TRAIN WITH REAL KAGGLE DATA - Final Thesis Models
Auto-deletes old results before saving new ones!
Run:  py -3.12 train_with_real_data.py
"""

import os, json, shutil, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ===== AUTO-CLEAN OLD RESULTS =====
header("CLEANING OLD RESULTS")

if os.path.exists("models/real_data"):
    shutil.rmtree("models/real_data")
    print("  Deleted: models/real_data/ (old JSON)")

if os.path.exists("charts"):
    shutil.rmtree("charts")
    print("  Deleted: charts/ (old images)")

os.makedirs("models/real_data", exist_ok=True)
os.makedirs("charts", exist_ok=True)
print("  Created fresh folders!")


# ===== FILE PATHS =====
BASE = os.path.join("data", "kaggle")
HAR_TRAIN = os.path.join(BASE, "human-activity-recognition-with-smartphones", "train.csv")
HAR_TEST  = os.path.join(BASE, "human-activity-recognition-with-smartphones", "test.csv")
OCC_TRAIN = os.path.join(BASE, "occupancy-detection-data-set-uci", "datatraining.txt")
OCC_TEST  = os.path.join(BASE, "occupancy-detection-data-set-uci", "datatest.txt")
EDU_DATA  = os.path.join(BASE, "xAPI-Edu-Data", "xAPI-Edu-Data.csv")

header("CHECKING FILES")
for name, path in [("HAR train", HAR_TRAIN), ("HAR test", HAR_TEST),
                     ("Occupancy train", OCC_TRAIN), ("Occupancy test", OCC_TEST),
                     ("xAPI-Edu", EDU_DATA)]:
    exists = "YES" if os.path.exists(path) else "NO"
    print(f"  {exists} {name}: {path}")

all_results = {}


# ================================================================
#  MODEL 1: ACTIVITY RECOGNITION - UCI HAR
# ================================================================
header("MODEL 1: Activity Recognition - UCI HAR (REAL!)")

if os.path.exists(HAR_TRAIN):
    har_train = pd.read_csv(HAR_TRAIN)
    print(f"  Train: {har_train.shape[0]} rows x {har_train.shape[1]} columns")

    har_test_df = None
    if os.path.exists(HAR_TEST):
        har_test_df = pd.read_csv(HAR_TEST)
        print(f"  Test:  {har_test_df.shape[0]} rows")
        print(f"  Total: {har_train.shape[0] + har_test_df.shape[0]} samples!")

    label_col = 'Activity'
    if label_col not in har_train.columns:
        label_col = har_train.columns[-1]

    print(f"\n  Label: '{label_col}'")
    print(f"  Distribution:\n{har_train[label_col].value_counts().to_string()}")

    exclude = [label_col]
    if 'subject' in har_train.columns:
        exclude.append('subject')
    feature_cols = [c for c in har_train.columns if c not in exclude]
    print(f"  Features: {len(feature_cols)}")

    le_har = LabelEncoder()
    X_train = np.nan_to_num(har_train[feature_cols].values, nan=0.0)
    y_train = le_har.fit_transform(har_train[label_col].values)
    class_names = list(le_har.classes_)

    if har_test_df is not None:
        X_test = np.nan_to_num(har_test_df[feature_cols].values, nan=0.0)
        y_test = le_har.transform(har_test_df[label_col].values)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "SVM (RBF)":          SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        "KNN (k=7)":          KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    har_scores = {}
    best_acc, best_name, best_pred = 0, "", None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        har_scores[name] = round(acc, 4)
        print(f"     Accuracy: {acc:.4f} ({acc:.2%}) | F1: {f1:.4f}")
        if acc > best_acc:
            best_acc, best_name, best_pred = acc, name, y_pred

    print(f"\n  BEST: {best_name} -> {best_acc:.2%}")
    print(f"\n  Classification Report:\n{classification_report(y_test, best_pred, target_names=class_names, zero_division=0)}")
    cm = confusion_matrix(y_test, best_pred).tolist()

    all_results['activity_recognition'] = {
        'dataset': 'UCI HAR (Kaggle)',
        'train_samples': len(X_train_s),
        'test_samples': len(X_test_s),
        'features': len(feature_cols),
        'classes': class_names,
        'scores': har_scores,
        'best_model': best_name,
        'best_accuracy': round(best_acc, 4),
        'confusion_matrix': cm,
    }


# ================================================================
#  MODEL 2: OCCUPANCY - PIR SENSOR
# ================================================================
header("MODEL 2: Occupancy Detection - PIR Sensor (REAL!)")

if os.path.exists(OCC_TRAIN):
    occ_train = pd.read_csv(OCC_TRAIN)
    print(f"  Train: {occ_train.shape[0]} rows | Columns: {list(occ_train.columns)}")

    occ_test_df = None
    if os.path.exists(OCC_TEST):
        occ_test_df = pd.read_csv(OCC_TEST)
        print(f"  Test:  {occ_test_df.shape[0]} rows")

    feature_cols = [c for c in occ_train.columns
                    if c != 'Occupancy'
                    and occ_train[c].dtype in ['float64','int64','float32','int32']
                    and 'date' not in c.lower()]

    X_train = np.nan_to_num(occ_train[feature_cols].values, nan=0.0)
    y_train = occ_train['Occupancy'].astype(int).values

    if occ_test_df is not None:
        X_test = np.nan_to_num(occ_test_df[feature_cols].values, nan=0.0)
        y_test = occ_test_df['Occupancy'].astype(int).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)

    occ_models = {
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM":                SVC(kernel='rbf', C=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "KNN":                KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    occ_scores = {}
    best_occ_acc, best_occ_name, best_occ_pred = 0, "", None

    for name, model in occ_models.items():
        print(f"\n  Training {name}...")
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
        acc = accuracy_score(y_test, y_pred)
        occ_scores[name] = round(acc, 4)
        print(f"     Accuracy: {acc:.4f} ({acc:.2%})")
        if acc > best_occ_acc:
            best_occ_acc, best_occ_name, best_occ_pred = acc, name, y_pred

    # Isolation Forest
    print(f"\n  Training Isolation Forest...")
    iso = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    iso.fit(X_tr_s)
    y_iso = np.array([1 if p == -1 else 0 for p in iso.predict(X_te_s)])
    occ_scores['Isolation Forest'] = round(accuracy_score(y_test, y_iso), 4)
    print(f"     Accuracy: {occ_scores['Isolation Forest']}")

    print(f"\n  BEST: {best_occ_name} -> {best_occ_acc:.2%}")
    print(f"\n  Classification Report:\n{classification_report(y_test, best_occ_pred, target_names=['Empty','Occupied'], zero_division=0)}")
    cm_occ = confusion_matrix(y_test, best_occ_pred).tolist()

    all_results['occupancy_detection'] = {
        'dataset': 'UCI Occupancy (Kaggle)',
        'train_samples': len(X_tr_s),
        'test_samples': len(X_te_s),
        'features': feature_cols,
        'scores': occ_scores,
        'best_model': best_occ_name,
        'best_accuracy': round(best_occ_acc, 4),
        'confusion_matrix': cm_occ,
    }


# ================================================================
#  MODEL 3: STUDENT BEHAVIOUR - xAPI-Edu-Data
# ================================================================
header("MODEL 3: Student Behaviour - xAPI-Edu-Data (REAL!)")

if os.path.exists(EDU_DATA):
    edu_df = pd.read_csv(EDU_DATA)
    print(f"  Loaded: {edu_df.shape[0]} rows x {edu_df.shape[1]} columns")

    label_col = 'Class'
    if label_col not in edu_df.columns:
        label_col = edu_df.columns[-1]

    print(f"  Label: '{label_col}'\n  Distribution:\n{edu_df[label_col].value_counts().to_string()}")

    le_label = LabelEncoder()
    y_all = le_label.fit_transform(edu_df[label_col].astype(str))
    class_labels = list(le_label.classes_)

    edu_encoded = edu_df.copy()
    for col in edu_encoded.columns:
        if col == label_col:
            continue
        if edu_encoded[col].dtype == 'object':
            edu_encoded[col] = LabelEncoder().fit_transform(edu_encoded[col].astype(str))

    feat_cols = [c for c in edu_encoded.columns if c != label_col]
    for col in feat_cols:
        edu_encoded[col] = pd.to_numeric(edu_encoded[col], errors='coerce')
    edu_encoded = edu_encoded.fillna(0)

    X = edu_encoded[feat_cols].values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all)

    edu_models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        "SVM":               SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        "KNN":               KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree":     DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    edu_scores = {}
    best_edu_acc, best_edu_name, best_edu_pred = 0, "", None

    for name, model in edu_models.items():
        print(f"\n  Training {name}...")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        edu_scores[name] = round(acc, 4)
        print(f"     Accuracy: {acc:.4f} ({acc:.2%})")
        if acc > best_edu_acc:
            best_edu_acc, best_edu_name, best_edu_pred = acc, name, y_pred

    print(f"\n  BEST: {best_edu_name} -> {best_edu_acc:.2%}")
    print(f"\n  Classification Report:\n{classification_report(y_te, best_edu_pred, target_names=[str(c) for c in class_labels], zero_division=0)}")
    cm_edu = confusion_matrix(y_te, best_edu_pred).tolist()

    all_results['student_behaviour'] = {
        'dataset': 'xAPI-Edu-Data (Kaggle)',
        'samples': len(edu_df),
        'features': len(feat_cols),
        'classes': [str(c) for c in class_labels],
        'scores': edu_scores,
        'best_model': best_edu_name,
        'best_accuracy': round(best_edu_acc, 4),
        'confusion_matrix': cm_edu,
    }


# ================================================================
#  SAVE FRESH JSON
# ================================================================
with open("models/real_data/kaggle_results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n  NEW results saved -> models/real_data/kaggle_results.json")


# ================================================================
#  AUTO-GENERATE CHARTS (from fresh JSON)
# ================================================================
header("AUTO-GENERATING CHARTS")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CHART 1: Bar Chart ---
print("  [1/4] Accuracy Comparison Bar Charts...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

tasks_info = [
    ('activity_recognition', 'Activity Recognition\n(UCI HAR)', '#3498db', (80, 100)),
    ('occupancy_detection', 'Occupancy Detection\n(PIR Sensor)', '#e74c3c', (80, 100)),
    ('student_behaviour', 'Student Behaviour\n(xAPI-Edu)', '#9b59b6', (40, 80)),
]

for idx, (key, title, color, ylim) in enumerate(tasks_info):
    if key not in all_results:
        continue
    data = all_results[key]
    names = list(data['scores'].keys())
    scores = [v * 100 for v in data['scores'].values()]
    best_s = data['best_accuracy'] * 100

    colors = ['#2ecc71' if s == max(scores) else color for s in scores]
    bars = axes[idx].bar(range(len(names)), scores, color=colors, edgecolor='black', linewidth=0.5)
    axes[idx].set_xticks(range(len(names)))
    axes[idx].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Accuracy (%)')
    axes[idx].set_ylim(ylim)
    axes[idx].axhline(y=best_s, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_s:.1f}%')
    axes[idx].legend(fontsize=9)
    for bar, val in zip(bars, scores):
        axes[idx].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.3,
                      f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/chart1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    SAVED: charts/chart1_accuracy_comparison.png")

# --- CHART 2: Best Models Horizontal Bar ---
print("  [2/4] Best Models Summary...")
fig, ax = plt.subplots(figsize=(10, 5))
labels, accs, colors = [], [], ['#3498db','#2ecc71','#e74c3c']
for k in ['activity_recognition','occupancy_detection','student_behaviour']:
    if k in all_results:
        d = all_results[k]
        labels.append(f"{d['best_model']}\n({k.replace('_',' ').title()})")
        accs.append(d['best_accuracy']*100)

bars = ax.barh(range(len(labels)), accs, color=colors[:len(labels)], edgecolor='black', height=0.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Accuracy (%)')
ax.set_title('Best Model Per Task (Real Kaggle Data)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 110)
for bar, val in zip(bars, accs):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2., f'{val:.2f}%', va='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/chart2_best_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("    SAVED: charts/chart2_best_models.png")

# --- CHART 3: Confusion Matrices ---
print("  [3/4] Confusion Matrices...")
cm_tasks = []
for k in ['activity_recognition','occupancy_detection','student_behaviour']:
    if k in all_results and 'confusion_matrix' in all_results[k]:
        cm_tasks.append((k, all_results[k]))

if len(cm_tasks) > 0:
    fig, axes = plt.subplots(1, len(cm_tasks), figsize=(6*len(cm_tasks), 5))
    if len(cm_tasks) == 1:
        axes = [axes]
    
    cmaps = ['Blues', 'Greens', 'Purples']
    for idx, (key, data) in enumerate(cm_tasks):
        cm = np.array(data['confusion_matrix'])
        classes = data.get('classes', [str(i) for i in range(cm.shape[0])])
        best = data['best_model']
        acc = data['best_accuracy']*100

        im = axes[idx].imshow(cm, cmap=cmaps[idx], interpolation='nearest')
        axes[idx].set_title(f"{key.replace('_',' ').title()}\n({best}, {acc:.1f}%)", fontsize=11, fontweight='bold')
        axes[idx].set_xticks(range(len(classes)))
        axes[idx].set_yticks(range(len(classes)))
        
        short_classes = [c[:8] for c in classes]
        axes[idx].set_xticklabels(short_classes, rotation=45, ha='right', fontsize=8)
        axes[idx].set_yticklabels(short_classes, fontsize=8)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, str(cm[i,j]), ha='center', va='center',
                             fontsize=10, fontweight='bold',
                             color='white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig('charts/chart3_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    SAVED: charts/chart3_confusion_matrices.png")

# --- CHART 4: Heatmap ---
print("  [4/4] Heatmap All Models...")
all_model_names = sorted(set(m for r in all_results.values() for m in r['scores']))
task_names = [k.replace('_',' ').title() for k in all_results]
matrix = []
for k in all_results:
    row = [all_results[k]['scores'].get(m, 0)*100 for m in all_model_names]
    matrix.append(row)

fig, ax = plt.subplots(figsize=(12, 4))
arr = np.array(matrix)
im = ax.imshow(arr, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
ax.set_xticks(range(len(all_model_names)))
ax.set_xticklabels(all_model_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(task_names)))
ax.set_yticklabels(task_names, fontsize=11)
ax.set_title('All Models x All Tasks — Accuracy Heatmap', fontsize=14, fontweight='bold')
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i,j] > 0:
            ax.text(j, i, f'{arr[i,j]:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white' if arr[i,j]>70 else 'black')
fig.colorbar(im, ax=ax, label='Accuracy (%)')
plt.tight_layout()
plt.savefig('charts/chart4_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("    SAVED: charts/chart4_heatmap.png")


# ================================================================
header("ALL DONE!")
print("""
  Old results DELETED + New results SAVED!

  JSON:   models/real_data/kaggle_results.json
  
  Charts: charts/chart1_accuracy_comparison.png
          charts/chart2_best_models.png
          charts/chart3_confusion_matrices.png
          charts/chart4_heatmap.png

  -> Open charts/ folder -> Insert into Word/Paper!
""")
