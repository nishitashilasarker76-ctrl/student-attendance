"""
DEEP ANALYSIS — Extract everything for thesis paper
Run: py -3.12 deep_analysis.py
Uses saved JSON + re-trains best models for extra metrics
"""

import os, json, warnings, time
warnings.filterwarnings('ignore')
start = time.time()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve, auc,
                             precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

os.makedirs("charts", exist_ok=True)

def header(t):
    print(f"\n{'='*60}")
    print(f"  {t}")
    print(f"{'='*60}")

B = os.path.join("data", "kaggle")
HAR_TR = os.path.join(B,"human-activity-recognition-with-smartphones","train.csv")
HAR_TE = os.path.join(B,"human-activity-recognition-with-smartphones","test.csv")
OCC_TR = os.path.join(B,"occupancy-detection-data-set-uci","datatraining.txt")
OCC_TE = os.path.join(B,"occupancy-detection-data-set-uci","datatest.txt")
EMP_DATA = None
for r,d,f in os.walk(B):
    for fn in f:
        if 'employee' in fn.lower() and fn.endswith('.csv'):
            EMP_DATA = os.path.join(r,fn); break

deep = {}

# ================================================================
#  1. DATASET STATISTICS TABLE
# ================================================================
header("1. DATASET STATISTICS")

datasets_info = []

# HAR
tr = pd.read_csv(HAR_TR); te = pd.read_csv(HAR_TE)
datasets_info.append({
    'Dataset': 'UCI HAR', 'Task': 'Activity Recognition',
    'Train': tr.shape[0], 'Test': te.shape[0], 'Total': tr.shape[0]+te.shape[0],
    'Features': tr.shape[1]-2, 'Classes': tr['Activity'].nunique(),
    'Source': 'UCI/Kaggle'
})

# Occupancy
otr = pd.read_csv(OCC_TR); ote = pd.read_csv(OCC_TE)
datasets_info.append({
    'Dataset': 'UCI Occupancy', 'Task': 'Presence Detection',
    'Train': otr.shape[0], 'Test': ote.shape[0], 'Total': otr.shape[0]+ote.shape[0],
    'Features': 5, 'Classes': 2, 'Source': 'UCI/Kaggle'
})

# Employee
emp = pd.read_csv(EMP_DATA)
datasets_info.append({
    'Dataset': 'Employee Eval', 'Task': 'Performance Classification',
    'Train': int(emp.shape[0]*0.8), 'Test': int(emp.shape[0]*0.2), 'Total': emp.shape[0],
    'Features': 14, 'Classes': emp['Performance_Label'].nunique(),
    'Source': 'Kaggle'
})

ds_df = pd.DataFrame(datasets_info)
print(ds_df.to_string(index=False))
deep['dataset_stats'] = datasets_info

# ================================================================
#  2. CHART: Dataset Class Distribution
# ================================================================
header("2. CLASS DISTRIBUTION CHARTS")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# HAR
har_counts = tr['Activity'].value_counts()
axes[0].barh(har_counts.index, har_counts.values, color=sns.color_palette("Set2", len(har_counts)))
axes[0].set_title('UCI HAR\nActivity Distribution', fontweight='bold')
axes[0].set_xlabel('Samples')
for i, v in enumerate(har_counts.values):
    axes[0].text(v+20, i, str(v), va='center', fontweight='bold')

# Occupancy
occ_counts = otr['Occupancy'].value_counts()
axes[1].bar(['Empty (0)', 'Occupied (1)'], occ_counts.values, color=['#3498db', '#e74c3c'])
axes[1].set_title('UCI Occupancy\nClass Distribution', fontweight='bold')
axes[1].set_ylabel('Samples')
for i, v in enumerate(occ_counts.values):
    axes[1].text(i, v+50, str(v), ha='center', fontweight='bold')

# Employee
emp_counts = emp['Performance_Label'].value_counts()
axes[2].bar(emp_counts.index, emp_counts.values, color=['#f39c12', '#e74c3c', '#2ecc71'])
axes[2].set_title('Employee Performance\nLabel Distribution', fontweight='bold')
axes[2].set_ylabel('Samples')
for i, v in enumerate(emp_counts.values):
    axes[2].text(i, v+30, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/deep1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  SAVED: charts/deep1_class_distribution.png")

# ================================================================
#  3. CORRELATION HEATMAP (Employee Dataset)
# ================================================================
header("3. CORRELATION HEATMAP")

num_cols = [c for c in emp.columns if emp[c].dtype in ['float64','int64'] and c != 'Employee_ID']
corr = emp[num_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title('Employee Dataset — Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/deep2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  SAVED: charts/deep2_correlation_heatmap.png")

# ================================================================
#  4. RETRAIN BEST MODELS + CROSS VALIDATION
# ================================================================
header("4. CROSS-VALIDATION (5-Fold)")

# --- HAR ---
print("\n  HAR (Logistic Regression):")
ex = ['Activity']
if 'subject' in tr.columns: ex.append('subject')
fc = [c for c in tr.columns if c not in ex]
Xall = np.nan_to_num(pd.concat([tr,te])[fc].values)
le_h = LabelEncoder()
yall = le_h.fit_transform(pd.concat([tr,te])['Activity'])
sc = StandardScaler(); Xall_s = sc.fit_transform(Xall)

lr_har = LogisticRegression(max_iter=1000, random_state=42)
cv_har = cross_val_score(lr_har, Xall_s, yall, cv=5, scoring='accuracy')
print(f"    5-Fold CV: {cv_har}")
print(f"    Mean: {cv_har.mean():.4f} (+/- {cv_har.std():.4f})")
deep['har_cv'] = {'mean': round(cv_har.mean(),4), 'std': round(cv_har.std(),4), 'folds': cv_har.tolist()}

# --- Occupancy ---
print("\n  Occupancy (Logistic Regression):")
ofc = ['Temperature','Humidity','Light','CO2','HumidityRatio']
Xo = np.nan_to_num(pd.concat([otr,ote])[ofc].values)
yo = pd.concat([otr,ote])['Occupancy'].astype(int).values
Xo_s = StandardScaler().fit_transform(Xo)

lr_occ = LogisticRegression(max_iter=500, random_state=42)
cv_occ = cross_val_score(lr_occ, Xo_s, yo, cv=5, scoring='accuracy')
print(f"    5-Fold CV: {cv_occ}")
print(f"    Mean: {cv_occ.mean():.4f} (+/- {cv_occ.std():.4f})")
deep['occ_cv'] = {'mean': round(cv_occ.mean(),4), 'std': round(cv_occ.std(),4), 'folds': cv_occ.tolist()}

# --- Employee ---
print("\n  Employee (Gradient Boosting):")
le_e = LabelEncoder()
ye = le_e.fit_transform(emp['Performance_Label'].astype(str))
efc = [c for c in emp.columns if c not in ['Performance_Label','Employee_ID']]
enc = emp.copy()
for c in enc.columns:
    if enc[c].dtype == 'object' and c != 'Performance_Label':
        enc[c] = LabelEncoder().fit_transform(enc[c].astype(str))
Xe = enc[efc].values.astype(float)
Xe_s = StandardScaler().fit_transform(Xe)

gb_emp = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
cv_emp = cross_val_score(gb_emp, Xe_s, ye, cv=5, scoring='accuracy')
print(f"    5-Fold CV: {cv_emp}")
print(f"    Mean: {cv_emp.mean():.4f} (+/- {cv_emp.std():.4f})")
deep['emp_cv'] = {'mean': round(cv_emp.mean(),4), 'std': round(cv_emp.std(),4), 'folds': cv_emp.tolist()}

# CV Chart
fig, ax = plt.subplots(figsize=(10, 5))
cv_data = [cv_har, cv_occ, cv_emp]
labels = ['Activity\n(Logistic Reg)', 'Occupancy\n(Logistic Reg)', 'Performance\n(Gradient Boost)']
bp = ax.boxplot(cv_data, labels=labels, patch_artist=True)
colors = ['#3498db', '#2ecc71', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel('Accuracy')
ax.set_title('5-Fold Cross-Validation Results', fontsize=13, fontweight='bold')
ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.3)
for i, cv in enumerate(cv_data):
    ax.text(i+1, cv.mean()+0.005, f'μ={cv.mean():.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/deep3_cross_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  SAVED: charts/deep3_cross_validation.png")

# ================================================================
#  5. FULL FEATURE IMPORTANCE (Employee)
# ================================================================
header("5. FEATURE IMPORTANCE (Employee Dataset)")

Xtr_e, Xte_e, ytr_e, yte_e = train_test_split(Xe_s, ye, test_size=0.2, random_state=42, stratify=ye)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
gb.fit(Xtr_e, ytr_e)

imp = sorted(zip(efc, gb.feature_importances_), key=lambda x: -x[1])
print("  All features ranked:")
for fn, iv in imp:
    bar = '#' * int(iv * 50)
    print(f"    {fn:>25}: {iv:.4f} {bar}")

deep['feature_importance'] = {fn: round(float(iv),4) for fn,iv in imp}

fig, ax = plt.subplots(figsize=(10, 7))
fnames = [x[0] for x in imp]
fvals = [x[1] for x in imp]
colors = ['#e74c3c' if v > 0.1 else '#3498db' if v > 0.05 else '#95a5a6' for v in fvals]
ax.barh(range(len(fnames)), fvals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(fnames)))
ax.set_yticklabels(fnames, fontsize=10)
ax.set_xlabel('Importance Score')
ax.set_title('Employee Performance — Feature Importance\n(Gradient Boosting)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(fvals):
    ax.text(v+0.002, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/deep4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  SAVED: charts/deep4_feature_importance.png")

# ================================================================
#  6. DETAILED METRICS TABLE
# ================================================================
header("6. DETAILED METRICS TABLE")

# HAR
Xtr_h = np.nan_to_num(tr[fc].values); ytr_h = le_h.transform(tr['Activity'])
Xte_h = np.nan_to_num(te[fc].values); yte_h = le_h.transform(te['Activity'])
sc_h = StandardScaler(); Xtr_h = sc_h.fit_transform(Xtr_h); Xte_h = sc_h.transform(Xte_h)
lr_h = LogisticRegression(max_iter=1000, random_state=42)
lr_h.fit(Xtr_h, ytr_h); yp_h = lr_h.predict(Xte_h)

# Occupancy
ofc_list = ['Temperature','Humidity','Light','CO2','HumidityRatio']
Xtr_o = otr[ofc_list].values; ytr_o = otr['Occupancy'].astype(int).values
Xte_o = ote[ofc_list].values; yte_o = ote['Occupancy'].astype(int).values
sc_o = StandardScaler(); Xtr_o = sc_o.fit_transform(Xtr_o); Xte_o = sc_o.transform(Xte_o)
lr_o = LogisticRegression(max_iter=500, random_state=42)
lr_o.fit(Xtr_o, ytr_o); yp_o = lr_o.predict(Xte_o)

# Employee
yp_e = gb.predict(Xte_e)

print("\n  MODEL 1: Employee Activity (HAR)")
print(f"    Accuracy:  {accuracy_score(yte_h, yp_h):.4f}")
print(f"    Precision: {precision_score(yte_h, yp_h, average='weighted'):.4f}")
print(f"    Recall:    {recall_score(yte_h, yp_h, average='weighted'):.4f}")
print(f"    F1-Score:  {f1_score(yte_h, yp_h, average='weighted'):.4f}")
print(f"    CV Mean:   {cv_har.mean():.4f} (+/-{cv_har.std():.4f})")

print("\n  MODEL 2: Office Occupancy")
print(f"    Accuracy:  {accuracy_score(yte_o, yp_o):.4f}")
print(f"    Precision: {precision_score(yte_o, yp_o, average='weighted'):.4f}")
print(f"    Recall:    {recall_score(yte_o, yp_o, average='weighted'):.4f}")
print(f"    F1-Score:  {f1_score(yte_o, yp_o, average='weighted'):.4f}")
print(f"    CV Mean:   {cv_occ.mean():.4f} (+/-{cv_occ.std():.4f})")

print("\n  MODEL 3: Employee Performance")
print(f"    Accuracy:  {accuracy_score(yte_e, yp_e):.4f}")
print(f"    Precision: {precision_score(yte_e, yp_e, average='weighted'):.4f}")
print(f"    Recall:    {recall_score(yte_e, yp_e, average='weighted'):.4f}")
print(f"    F1-Score:  {f1_score(yte_e, yp_e, average='weighted'):.4f}")
print(f"    CV Mean:   {cv_emp.mean():.4f} (+/-{cv_emp.std():.4f})")

deep['detailed_metrics'] = {
    'activity': {
        'accuracy': round(accuracy_score(yte_h,yp_h),4),
        'precision': round(precision_score(yte_h,yp_h,average='weighted'),4),
        'recall': round(recall_score(yte_h,yp_h,average='weighted'),4),
        'f1': round(f1_score(yte_h,yp_h,average='weighted'),4),
    },
    'occupancy': {
        'accuracy': round(accuracy_score(yte_o,yp_o),4),
        'precision': round(precision_score(yte_o,yp_o,average='weighted'),4),
        'recall': round(recall_score(yte_o,yp_o,average='weighted'),4),
        'f1': round(f1_score(yte_o,yp_o,average='weighted'),4),
    },
    'performance': {
        'accuracy': round(accuracy_score(yte_e,yp_e),4),
        'precision': round(precision_score(yte_e,yp_e,average='weighted'),4),
        'recall': round(recall_score(yte_e,yp_e,average='weighted'),4),
        'f1': round(f1_score(yte_e,yp_e,average='weighted'),4),
    }
}

# Metrics comparison chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_names))
w = 0.25

for task, dm in deep['detailed_metrics'].items():
    vals = [dm['accuracy'], dm['precision'], dm['recall'], dm['f1']]
    print(f"  {task}: {vals}")

v1 = [deep['detailed_metrics']['activity'][m] for m in ['accuracy','precision','recall','f1']]
v2 = [deep['detailed_metrics']['occupancy'][m] for m in ['accuracy','precision','recall','f1']]
v3 = [deep['detailed_metrics']['performance'][m] for m in ['accuracy','precision','recall','f1']]

bars1 = ax.bar(x-w, [v*100 for v in v1], w, label='Activity (HAR)', color='#3498db', edgecolor='black')
bars2 = ax.bar(x, [v*100 for v in v2], w, label='Occupancy (PIR)', color='#2ecc71', edgecolor='black')
bars3 = ax.bar(x+w, [v*100 for v in v3], w, label='Performance (Emp)', color='#e74c3c', edgecolor='black')

ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_ylim(85, 100)
ax.set_title('Detailed Metrics Comparison — All Tasks', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

for bars in [bars1, bars2, bars3]:
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.2, f'{b.get_height():.1f}',
                ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/deep5_detailed_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  SAVED: charts/deep5_detailed_metrics.png")

# ================================================================
#  7. EMPLOYEE DATA ANALYSIS CHARTS
# ================================================================
header("7. EMPLOYEE DATA ANALYSIS")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Attendance vs Performance
axes[0,0].scatter(emp[emp['Performance_Label']=='High']['Attendance_Rate'],
                  emp[emp['Performance_Label']=='High']['Productivity_Index'],
                  c='#2ecc71', label='High', alpha=0.6, s=20)
axes[0,0].scatter(emp[emp['Performance_Label']=='Medium']['Attendance_Rate'],
                  emp[emp['Performance_Label']=='Medium']['Productivity_Index'],
                  c='#f39c12', label='Medium', alpha=0.3, s=10)
axes[0,0].scatter(emp[emp['Performance_Label']=='Low']['Attendance_Rate'],
                  emp[emp['Performance_Label']=='Low']['Productivity_Index'],
                  c='#e74c3c', label='Low', alpha=0.3, s=10)
axes[0,0].set_xlabel('Attendance Rate (%)'); axes[0,0].set_ylabel('Productivity Index')
axes[0,0].set_title('Attendance vs Productivity', fontweight='bold')
axes[0,0].legend()

# Late Arrivals distribution
for label, color in [('High','#2ecc71'),('Medium','#f39c12'),('Low','#e74c3c')]:
    subset = emp[emp['Performance_Label']==label]['Late_Arrivals']
    axes[0,1].hist(subset, bins=20, alpha=0.5, label=label, color=color)
axes[0,1].set_xlabel('Late Arrivals'); axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Late Arrivals by Performance', fontweight='bold')
axes[0,1].legend()

# Tasks Completed vs Assigned
axes[1,0].scatter(emp['Tasks_Assigned'], emp['Tasks_Completed'],
                  c=emp['Productivity_Index'], cmap='RdYlGn', alpha=0.4, s=10)
axes[1,0].set_xlabel('Tasks Assigned'); axes[1,0].set_ylabel('Tasks Completed')
axes[1,0].set_title('Tasks Assigned vs Completed', fontweight='bold')
axes[1,0].plot([0,emp['Tasks_Assigned'].max()],[0,emp['Tasks_Assigned'].max()],'r--',alpha=0.3)

# Supervisor Rating distribution by performance
data_sup = [emp[emp['Performance_Label']=='High']['Supervisor_Rating'],
            emp[emp['Performance_Label']=='Medium']['Supervisor_Rating'],
            emp[emp['Performance_Label']=='Low']['Supervisor_Rating']]
bp = axes[1,1].boxplot(data_sup, labels=['High','Medium','Low'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#2ecc71','#f39c12','#e74c3c']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1,1].set_ylabel('Supervisor Rating')
axes[1,1].set_title('Supervisor Rating by Performance', fontweight='bold')

plt.suptitle('Employee Data Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/deep6_employee_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  SAVED: charts/deep6_employee_analysis.png")

# ================================================================
#  SAVE ALL DEEP ANALYSIS
# ================================================================
with open("models/real_data/deep_analysis.json", 'w') as f:
    json.dump(deep, f, indent=2, default=str)

total = time.time() - start
header(f"ALL DONE! Time: {total:.0f} seconds")
print(f"""
  CHARTS GENERATED (for thesis paper):
  
  charts/deep1_class_distribution.png    - Dataset class distribution
  charts/deep2_correlation_heatmap.png   - Feature correlation (Employee)
  charts/deep3_cross_validation.png      - 5-Fold CV box plot
  charts/deep4_feature_importance.png    - Full feature importance
  charts/deep5_detailed_metrics.png      - Accuracy/Precision/Recall/F1
  charts/deep6_employee_analysis.png     - Employee data scatter/box plots
  
  + Previous charts:
  charts/chart1_accuracy_comparison.png
  charts/chart2_confusion_matrices.png  
  charts/chart3_best_models.png
  
  JSON: models/real_data/deep_analysis.json
  
  Total charts: 9 images -> Insert into thesis paper!
""")
