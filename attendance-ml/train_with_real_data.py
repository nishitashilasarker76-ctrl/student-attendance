"""
EMPLOYEE ATTENDANCE & ACTIVITY MONITORING SYSTEM
Train with REAL Kaggle Data — 3 Datasets (Employee focused!)
Auto-cleans old results + Auto-generates charts
Run: py -3.12 train_with_real_data.py
"""

import os
import json
import shutil
import warnings
import time
warnings.filterwarnings('ignore')

START_TIME = time.time()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, IsolationForest
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ===== CONSTANTS =====
TASK_TITLES = {
    "employee_activity": "Employee Activity\nRecognition",
    "office_occupancy": "Office Occupancy\nDetection",
    "employee_performance": "Employee Performance\nClassification",
}

TASK_COLORS = {
    "employee_activity": "#3498db",
    "office_occupancy": "#2ecc71",
    "employee_performance": "#e74c3c",
}

TASK_YLIMS = {
    "employee_activity": (80, 100),
    "office_occupancy": (80, 100),
    "employee_performance": (40, 100),
}


def header(t):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {t}")
    print(f"{'='*60}")


def train_and_score(models_dict, Xtr, ytr, Xte, yte, sample_weights=None):
    """
    Train multiple models and return best model name, best accuracy, predictions, and scores dict.
    
    Args:
        models_dict: Dict of {name: sklearn_model}
        Xtr, ytr: Training data
        Xte, yte: Test data
        sample_weights: Optional sample weights for training
        
    Returns:
        (best_model_name, best_accuracy, best_predictions, scores_dict)
    """
    scores = {}
    best_acc = 0
    best_name = ""
    best_pred = None
    
    for name, model in models_dict.items():
        print(f"  Training {name}...", end="", flush=True)
        
        if sample_weights is not None:
            model.fit(Xtr, ytr, sample_weight=sample_weights)
        else:
            model.fit(Xtr, ytr)
        
        yp = model.predict(Xte)
        acc = accuracy_score(yte, yp)
        scores[name] = round(acc, 4)
        
        print(f" {acc:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pred = yp
    
    return best_name, best_acc, best_pred, scores


def train_isolation_forest(Xtr, Xte, yte, contamination=0.15):
    """
    Train Isolation Forest and return accuracy and predictions.
    
    Args:
        Xtr, Xte: Training and test features
        yte: Test labels
        contamination: Contamination parameter
        
    Returns:
        (accuracy, predictions)
    """
    print(f"  Training Isolation Forest...", end="", flush=True)
    iso = IsolationForest(
        n_estimators=100, 
        contamination=contamination, 
        random_state=42
    )
    iso.fit(Xtr)
    yi = np.array([1 if p == -1 else 0 for p in iso.predict(Xte)])
    acc = accuracy_score(yte, yi)
    print(f" {acc:.2%}")
    return acc, yi

# ===== AUTO-CLEAN =====
header("CLEANING OLD RESULTS")
for d in ["models/real_data","charts"]:
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d,exist_ok=True)
print("  Fresh folders created!")

# ===== PATHS =====
B = os.path.join("data","kaggle")
HAR_TR = os.path.join(B,"human-activity-recognition-with-smartphones","train.csv")
HAR_TE = os.path.join(B,"human-activity-recognition-with-smartphones","test.csv")
OCC_TR = os.path.join(B,"occupancy-detection-data-set-uci","datatraining.txt")
OCC_TE = os.path.join(B,"occupancy-detection-data-set-uci","datatest.txt")

# Employee dataset - search for it
EMP_DATA = None
for root,dirs,files in os.walk(B):
    for f in files:
        if 'employee' in f.lower() and f.endswith('.csv'):
            EMP_DATA = os.path.join(root,f)
            break

header("CHECKING FILES")
for n,p in [("HAR train",HAR_TR),("HAR test",HAR_TE),("Occupancy",OCC_TR),("Employee",EMP_DATA or "NOT FOUND")]:
    e = "YES" if p and os.path.exists(p) else "NO"
    print(f"  {e} {n}: {p}")

R = {}

# ================================================================
#  MODEL 1: EMPLOYEE ACTIVITY RECOGNITION — UCI HAR
# ================================================================
header("MODEL 1: Employee Activity Recognition (UCI HAR)")

if os.path.exists(HAR_TR):
    t1=time.time()
    tr=pd.read_csv(HAR_TR); te=pd.read_csv(HAR_TE)
    print(f"  Train:{tr.shape[0]} Test:{te.shape[0]} Total:{tr.shape[0]+te.shape[0]}")

    lc='Activity'
    if lc not in tr.columns: lc=tr.columns[-1]
    print(f"  Label:'{lc}' Classes:{tr[lc].nunique()}")

    ex=[lc]
    if 'subject' in tr.columns: ex.append('subject')
    fc=[c for c in tr.columns if c not in ex]

    le=LabelEncoder()
    Xtr=np.nan_to_num(tr[fc].values); ytr=le.fit_transform(tr[lc])
    Xte=np.nan_to_num(te[fc].values); yte=le.transform(te[lc])
    cn=list(le.classes_)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Logistic Reg": LogisticRegression(max_iter=1000, random_state=42),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }

    best_name, best_acc, best_pred, scores = train_and_score(
        models, Xtr, ytr, Xte, yte
    )

    print(f"\n  BEST: {best_name} -> {best_acc:.2%}")
    print(classification_report(yte, best_pred, target_names=cn, zero_division=0))
    cm = confusion_matrix(yte, best_pred).tolist()

    R['employee_activity'] = {
        'dataset': 'UCI HAR (Kaggle)',
        'samples': tr.shape[0] + te.shape[0],
        'features': len(fc),
        'classes': cn,
        'scores': scores,
        'best_model': best_name,
        'best_accuracy': round(best_acc, 4),
        'confusion_matrix': cm
    }
    print(f"  Time: {time.time()-t1:.1f}s")

# ================================================================
#  MODEL 2: OFFICE OCCUPANCY DETECTION — PIR SENSOR
# ================================================================
header("MODEL 2: Office Occupancy Detection (PIR Sensor)")

if os.path.exists(OCC_TR):
    t1=time.time()
    otr=pd.read_csv(OCC_TR); ote=pd.read_csv(OCC_TE)
    print(f"  Train:{otr.shape[0]} Test:{ote.shape[0]}")

    fc = [
        c for c in otr.columns
        if c != 'Occupancy'
        and otr[c].dtype in ['float64', 'int64']
        and 'date' not in c.lower()
    ]
    print(f"  Features: {fc}")

    Xtr = np.nan_to_num(otr[fc].values)
    ytr = otr['Occupancy'].astype(int).values
    Xte = np.nan_to_num(ote[fc].values)
    yte = ote['Occupancy'].astype(int).values

    s = StandardScaler()
    Xtr = s.fit_transform(Xtr)
    Xte = s.transform(Xte)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Logistic Reg": LogisticRegression(max_iter=500, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    best_name, best_acc, best_pred, scores = train_and_score(
        models, Xtr, ytr, Xte, yte
    )

    # Isolation Forest
    iso_acc, iso_pred = train_isolation_forest(Xtr, Xte, yte, contamination=0.15)
    scores['Isolation Forest'] = round(iso_acc, 4)
    
    # Check if Isolation Forest is better
    if iso_acc > best_acc:
        best_acc = iso_acc
        best_name = 'Isolation Forest'
        best_pred = iso_pred

    print(f"\n  BEST: {best_name} -> {best_acc:.2%}")
    print(classification_report(
        yte, best_pred, 
        target_names=['Empty', 'Occupied'], 
        zero_division=0
    ))
    cm = confusion_matrix(yte, best_pred).tolist()

    R['office_occupancy'] = {
        'dataset': 'UCI Occupancy (Kaggle)',
        'samples': otr.shape[0] + ote.shape[0],
        'features': fc,
        'scores': scores,
        'best_model': best_name,
        'best_accuracy': round(best_acc, 4),
        'confusion_matrix': cm
    }
    print(f"  Time: {time.time()-t1:.1f}s")

# ================================================================
#  MODEL 3: EMPLOYEE PERFORMANCE — Attendance + Activity
# ================================================================
header("MODEL 3: Employee Performance & Attendance")

if EMP_DATA and os.path.exists(EMP_DATA):
    t1=time.time()
    df=pd.read_csv(EMP_DATA)
    print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # Find label column
    label_col = None
    for c in df.columns:
        cl = c.lower()
        if (
            'performance' in cl 
            and ('label' in cl or 'level' in cl or 'rating' in cl or 'category' in cl)
        ):
            label_col = c
            break
    if label_col is None:
        for c in df.columns:
            if 'performance' in c.lower():
                label_col = c
                break
    if label_col is None:
        label_col = df.columns[-1]

    print(f"\n  Label: '{label_col}'")
    print(f"  Distribution:\n{df[label_col].value_counts().to_string()}")

    # Encode label
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    classes = list(le.classes_)
    print(f"  Classes: {classes}")

    # Encode other object columns
    encoded_df = df.copy()
    for c in encoded_df.columns:
        if c == label_col:
            continue
        if encoded_df[c].dtype == 'object':
            encoded_df[c] = LabelEncoder().fit_transform(encoded_df[c].astype(str))

    # Features
    feature_cols = [
        c for c in encoded_df.columns
        if c != label_col and c.lower() != 'employee_id'
    ]
    for c in feature_cols:
        encoded_df[c] = pd.to_numeric(encoded_df[c], errors='coerce')
    encoded_df = encoded_df.fillna(0)

    X = encoded_df[feature_cols].values.astype(float)
    X = StandardScaler().fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Features:{len(feature_cols)} Train:{len(Xtr)} Test:{len(Xte)}")

    class_weight = 'balanced'
    sample_weight = compute_sample_weight(class_weight=class_weight, y=ytr)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, 
            class_weight=class_weight
        ),
        "Gradient Boost": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1, weights='distance'
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight=class_weight
        ),
    }

    scores = {}
    best_acc = 0
    best_name = ""
    best_pred = None
    best_model_obj = None
    
    for name, model in models.items():
        print(f"  Training {name}...", end="", flush=True)
        
        if name == "Gradient Boost":
            model.fit(Xtr, ytr, sample_weight=sample_weight)
        else:
            model.fit(Xtr, ytr)
        
        yp = model.predict(Xte)
        acc = accuracy_score(yte, yp)
        f1_macro = f1_score(yte, yp, average='macro')
        scores[name] = round(acc, 4)
        print(f" {acc:.2%} (macro-F1:{f1_macro:.4f})")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pred = yp
            best_model_obj = model

    print(f"\n  BEST: {best_name} -> {best_acc:.2%}")
    print(classification_report(
        yte, best_pred, 
        target_names=[str(c) for c in classes], 
        zero_division=0
    ))
    cm = confusion_matrix(yte, best_pred).tolist()

    # Feature importance
    if hasattr(best_model_obj, 'feature_importances_'):
        print("  Top 5 Features:")
        importances = sorted(
            zip(feature_cols, best_model_obj.feature_importances_),
            key=lambda x: -x[1]
        )
        for fname, fval in importances[:5]:
            print(f"    {fname:>25}: {fval:.4f} {'#' * int(fval * 40)}")

    R['employee_performance'] = {
        'dataset': 'Employee Activity & Evaluation (Kaggle)',
        'samples': len(df),
        'features': len(feature_cols),
        'classes': [str(c) for c in classes],
        'scores': scores,
        'best_model': best_name,
        'best_accuracy': round(best_acc, 4),
        'confusion_matrix': cm
    }
    print(f"  Time: {time.time()-t1:.1f}s")
else:
    print(f"  Employee dataset not found!")

# ================================================================
#  SAVE JSON
# ================================================================
with open("models/real_data/kaggle_results.json",'w') as f:
    json.dump(R,f,indent=2,default=str)
print(f"\n  Results saved -> models/real_data/kaggle_results.json")

# ================================================================
#  AUTO-GENERATE CHARTS
# ================================================================
header("GENERATING CHARTS")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CHART 1: Accuracy Bars
print("  [1/3] Accuracy comparison...")
fig, axes = plt.subplots(1, len(R), figsize=(6 * len(R), 6))
if len(R) == 1:
    axes = [axes]

for i, (task_key, task_data) in enumerate(R.items()):
    model_names = list(task_data['scores'].keys())
    accuracies = [v * 100 for v in task_data['scores'].values()]
    best_acc = task_data['best_accuracy'] * 100
    
    color = TASK_COLORS.get(task_key, '#3498db')
    ylim = TASK_YLIMS.get(task_key, (40, 100))
    bar_colors = [
        '#2ecc71' if acc == max(accuracies) else color 
        for acc in accuracies
    ]
    
    bars = axes[i].bar(
        range(len(model_names)), accuracies, color=bar_colors,
        edgecolor='black', linewidth=0.5
    )
    axes[i].set_xticks(range(len(model_names)))
    axes[i].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes[i].set_title(
        TASK_TITLES.get(task_key, task_key),
        fontsize=12, fontweight='bold'
    )
    axes[i].set_ylabel('Accuracy (%)')
    axes[i].set_ylim(ylim)
    axes[i].axhline(
        y=best_acc, color='red', linestyle='--', alpha=0.5,
        label=f'Best:{best_acc:.1f}%'
    )
    axes[i].legend(fontsize=8)
    
    for bar, acc in zip(bars, accuracies):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold'
        )

plt.tight_layout()
plt.savefig('charts/chart1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    SAVED: charts/chart1_accuracy_comparison.png")

# CHART 2: Confusion Matrices
print("  [2/3] Confusion matrices...")
cm_list = [(k, d) for k, d in R.items() if 'confusion_matrix' in d]
if cm_list:
    fig, axes = plt.subplots(1, len(cm_list), figsize=(5 * len(cm_list), 4.5))
    if len(cm_list) == 1:
        axes = [axes]
    cmaps = ['Blues', 'Greens', 'Reds']
    
    for i, (task_key, task_data) in enumerate(cm_list):
        cm = np.array(task_data['confusion_matrix'])
        classes = task_data.get('classes', [str(j) for j in range(cm.shape[0])])
        
        axes[i].imshow(cm, cmap=cmaps[i % 3], interpolation='nearest')
        axes[i].set_title(
            f"{TASK_TITLES.get(task_key, task_key)}\n"
            f"({task_data['best_model']}, {task_data['best_accuracy']*100:.1f}%)",
            fontsize=9, fontweight='bold'
        )
        
        class_labels = [c[:10] for c in classes]
        axes[i].set_xticks(range(len(class_labels)))
        axes[i].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(len(class_labels)))
        axes[i].set_yticklabels(class_labels, fontsize=7)
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')
        
        threshold = cm.max() / 2.
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                text_color = 'white' if cm[row, col] > threshold else 'black'
                axes[i].text(
                    col, row, str(cm[row, col]),
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color=text_color
                )
    
    plt.tight_layout()
    plt.savefig('charts/chart2_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    SAVED: charts/chart2_confusion_matrices.png")

# CHART 3: Summary
print("  [3/3] Summary chart...")
fig, ax = plt.subplots(figsize=(10, 5))
labels = []
accuracies = []
colors = ['#3498db', '#2ecc71', '#e74c3c']

for task_key in R:
    task_data = R[task_key]
    task_title = TASK_TITLES.get(task_key, task_key).replace('\n', ' ')
    labels.append(f"{task_data['best_model']}\n({task_title})")
    accuracies.append(task_data['best_accuracy'] * 100)

bars = ax.barh(
    range(len(labels)), accuracies, color=colors[:len(labels)],
    edgecolor='black', height=0.5
)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Accuracy (%)')
ax.set_title(
    'Best Model Per Task — Employee Monitoring System',
    fontsize=13, fontweight='bold'
)
ax.set_xlim(0, 110)

for bar, acc in zip(bars, accuracies):
    ax.text(
        acc + 0.5, bar.get_y() + bar.get_height() / 2.,
        f'{acc:.2f}%', va='center', fontsize=12, fontweight='bold'
    )

plt.tight_layout()
plt.savefig('charts/chart3_best_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("    SAVED: charts/chart3_best_models.png")

# ================================================================
#  GRAND SUMMARY
# ================================================================
total_time = time.time() - START_TIME
header(f"DONE! Total time: {total_time:.0f} seconds")

print("\n  RESULTS SUMMARY:")
for task_key, task_data in R.items():
    task_title = TASK_TITLES.get(task_key, task_key).replace('\n', ' ')
    print(f"\n  {task_title}")
    print(f"    Dataset: {task_data['dataset']} ({task_data.get('samples', '?')} samples)")
    print(f"    BEST: {task_data['best_model']} -> {task_data['best_accuracy']:.2%}")
    for model_name, score in task_data['scores'].items():
        is_best = " <-- BEST" if model_name == task_data['best_model'] else ""
        print(f"      {model_name:>20}: {score:.4f} ({score:.2%}){is_best}")

print(f"""
  FILES:
    models/real_data/kaggle_results.json
    charts/chart1_accuracy_comparison.png
    charts/chart2_confusion_matrices.png
    charts/chart3_best_models.png
""")

