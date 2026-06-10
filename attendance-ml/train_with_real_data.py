"""
EMPLOYEE ATTENDANCE & ACTIVITY MONITORING SYSTEM
Train with REAL Kaggle Data — 3 Datasets (Employee focused!)
Auto-cleans old results + Auto-generates charts
Run: py -3.12 train_with_real_data.py
"""

import os, json, shutil, warnings, time
warnings.filterwarnings('ignore')
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def header(t):
    print(f"\n{'='*60}")
    print(f"  {t}")
    print(f"{'='*60}")

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

    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)

    ms={
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
        "Logistic Reg":LogisticRegression(max_iter=1000,random_state=42),
        "KNN (k=7)":KNeighborsClassifier(n_neighbors=7,n_jobs=-1),
    }

    sc_d={}; ba=0; bn=""; bp=None
    for n,m in ms.items():
        print(f"  Training {n}...",end="",flush=True)
        m.fit(Xtr,ytr); yp=m.predict(Xte)
        a=accuracy_score(yte,yp); f1=f1_score(yte,yp,average='weighted')
        sc_d[n]=round(a,4)
        print(f" {a:.2%} (F1:{f1:.4f})")
        if a>ba: ba=a;bn=n;bp=yp

    print(f"\n  BEST: {bn} -> {ba:.2%}")
    print(classification_report(yte,bp,target_names=cn,zero_division=0))
    cm=confusion_matrix(yte,bp).tolist()

    R['employee_activity']={
        'dataset':'UCI HAR (Kaggle)','samples':tr.shape[0]+te.shape[0],
        'features':len(fc),'classes':cn,'scores':sc_d,
        'best_model':bn,'best_accuracy':round(ba,4),'confusion_matrix':cm
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

    fc=[c for c in otr.columns if c!='Occupancy' and otr[c].dtype in ['float64','int64'] and 'date' not in c.lower()]
    print(f"  Features: {fc}")

    Xtr=np.nan_to_num(otr[fc].values); ytr=otr['Occupancy'].astype(int).values
    Xte=np.nan_to_num(ote[fc].values); yte=ote['Occupancy'].astype(int).values

    s=StandardScaler(); Xtr=s.fit_transform(Xtr); Xte=s.transform(Xte)

    ms={
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
        "Logistic Reg":LogisticRegression(max_iter=500,random_state=42),
        "KNN":KNeighborsClassifier(n_neighbors=5,n_jobs=-1),
    }

    sc_d={}; ba=0; bn=""; bp=None
    for n,m in ms.items():
        print(f"  Training {n}...",end="",flush=True)
        m.fit(Xtr,ytr); yp=m.predict(Xte)
        a=accuracy_score(yte,yp); sc_d[n]=round(a,4)
        print(f" {a:.2%}")
        if a>ba: ba=a;bn=n;bp=yp

    # Isolation Forest
    print(f"  Training Isolation Forest...",end="",flush=True)
    iso=IsolationForest(n_estimators=100,contamination=0.15,random_state=42)
    iso.fit(Xtr); yi=np.array([1 if p==-1 else 0 for p in iso.predict(Xte)])
    ia=accuracy_score(yte,yi); sc_d['Isolation Forest']=round(ia,4)
    print(f" {ia:.2%}")

    print(f"\n  BEST: {bn} -> {ba:.2%}")
    print(classification_report(yte,bp,target_names=['Empty','Occupied'],zero_division=0))
    cm=confusion_matrix(yte,bp).tolist()

    R['office_occupancy']={
        'dataset':'UCI Occupancy (Kaggle)','samples':otr.shape[0]+ote.shape[0],
        'features':fc,'scores':sc_d,
        'best_model':bn,'best_accuracy':round(ba,4),'confusion_matrix':cm
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
    lc=None
    for c in df.columns:
        cl=c.lower()
        if 'performance' in cl and ('label' in cl or 'level' in cl or 'rating' in cl or 'category' in cl):
            lc=c; break
    if lc is None:
        for c in df.columns:
            if 'performance' in c.lower(): lc=c; break
    if lc is None:
        lc=df.columns[-1]

    print(f"\n  Label: '{lc}'")
    print(f"  Distribution:\n{df[lc].value_counts().to_string()}")

    # Encode label
    le=LabelEncoder()
    y=le.fit_transform(df[lc].astype(str))
    cl=list(le.classes_)
    print(f"  Classes: {cl}")

    # Encode other object columns
    enc=df.copy()
    for c in enc.columns:
        if c==lc: continue
        if enc[c].dtype=='object':
            enc[c]=LabelEncoder().fit_transform(enc[c].astype(str))

    # Features
    fc=[c for c in enc.columns if c!=lc and c.lower()!='employee_id']
    for c in fc:
        enc[c]=pd.to_numeric(enc[c],errors='coerce')
    enc=enc.fillna(0)

    X=enc[fc].values.astype(float)
    X=StandardScaler().fit_transform(X)

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print(f"  Features:{len(fc)} Train:{len(Xtr)} Test:{len(Xte)}")

    cw='balanced'
    sw=compute_sample_weight(class_weight=cw,y=ytr)

    ms={
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,class_weight=cw),
        "Gradient Boost":GradientBoostingClassifier(n_estimators=100,max_depth=4,random_state=42),
        "KNN":KNeighborsClassifier(n_neighbors=5,n_jobs=-1,weights='distance'),
        "Decision Tree":DecisionTreeClassifier(max_depth=8,random_state=42,class_weight=cw),
    }

    sc_d={}; ba=0; bn=""; bp=None
    for n,m in ms.items():
        print(f"  Training {n}...",end="",flush=True)
        if n=="Gradient Boost":
            m.fit(Xtr,ytr,sample_weight=sw)
        else:
            m.fit(Xtr,ytr)
        yp=m.predict(Xte)
        a=accuracy_score(yte,yp); f1m=f1_score(yte,yp,average='macro')
        sc_d[n]=round(a,4)
        print(f" {a:.2%} (macro-F1:{f1m:.4f})")
        if a>ba: ba=a;bn=n;bp=yp

    print(f"\n  BEST: {bn} -> {ba:.2%}")
    print(classification_report(yte,bp,target_names=[str(c) for c in cl],zero_division=0))
    cm=confusion_matrix(yte,bp).tolist()

    # Feature importance
    if hasattr(ms[bn],'feature_importances_'):
        print("  Top 5 Features:")
        imp=sorted(zip(fc,ms[bn].feature_importances_),key=lambda x:-x[1])
        for fn,iv in imp[:5]:
            print(f"    {fn:>25}: {iv:.4f} {'#'*int(iv*40)}")

    R['employee_performance']={
        'dataset':'Employee Activity & Evaluation (Kaggle)','samples':len(df),
        'features':len(fc),'classes':[str(c) for c in cl],'scores':sc_d,
        'best_model':bn,'best_accuracy':round(ba,4),'confusion_matrix':cm
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
fig,axes=plt.subplots(1,len(R),figsize=(6*len(R),6))
if len(R)==1: axes=[axes]

colors_map={'employee_activity':'#3498db','office_occupancy':'#2ecc71','employee_performance':'#e74c3c'}
titles_map={'employee_activity':'Employee Activity\nRecognition','office_occupancy':'Office Occupancy\nDetection','employee_performance':'Employee Performance\nClassification'}
ylims_map={'employee_activity':(80,100),'office_occupancy':(80,100),'employee_performance':(40,100)}

for i,(k,d) in enumerate(R.items()):
    names=list(d['scores'].keys()); vals=[v*100 for v in d['scores'].values()]
    best_v=d['best_accuracy']*100
    c=colors_map.get(k,'#3498db'); yl=ylims_map.get(k,(40,100))
    cs=['#2ecc71' if v==max(vals) else c for v in vals]
    bars=axes[i].bar(range(len(names)),vals,color=cs,edgecolor='black',linewidth=0.5)
    axes[i].set_xticks(range(len(names))); axes[i].set_xticklabels(names,rotation=45,ha='right',fontsize=8)
    axes[i].set_title(titles_map.get(k,k),fontsize=12,fontweight='bold')
    axes[i].set_ylabel('Accuracy (%)'); axes[i].set_ylim(yl)
    axes[i].axhline(y=best_v,color='red',linestyle='--',alpha=0.5,label=f'Best:{best_v:.1f}%')
    axes[i].legend(fontsize=8)
    for b,v in zip(bars,vals):
        axes[i].text(b.get_x()+b.get_width()/2.,b.get_height()+0.3,f'{v:.1f}%',ha='center',fontsize=9,fontweight='bold')

plt.tight_layout()
plt.savefig('charts/chart1_accuracy_comparison.png',dpi=300,bbox_inches='tight')
plt.close()
print("    SAVED: charts/chart1_accuracy_comparison.png")

# CHART 2: Confusion Matrices
print("  [2/3] Confusion matrices...")
cm_list=[(k,d) for k,d in R.items() if 'confusion_matrix' in d]
if cm_list:
    fig,axes=plt.subplots(1,len(cm_list),figsize=(5*len(cm_list),4.5))
    if len(cm_list)==1: axes=[axes]
    cmaps=['Blues','Greens','Reds']
    for i,(k,d) in enumerate(cm_list):
        cm=np.array(d['confusion_matrix']); cl=d.get('classes',[str(j) for j in range(cm.shape[0])])
        im=axes[i].imshow(cm,cmap=cmaps[i%3],interpolation='nearest')
        axes[i].set_title(f"{titles_map.get(k,k)}\n({d['best_model']},{d['best_accuracy']*100:.1f}%)",fontsize=9,fontweight='bold')
        sc=[c[:10] for c in cl]
        axes[i].set_xticks(range(len(sc))); axes[i].set_xticklabels(sc,rotation=45,ha='right',fontsize=7)
        axes[i].set_yticks(range(len(sc))); axes[i].set_yticklabels(sc,fontsize=7)
        axes[i].set_ylabel('Actual'); axes[i].set_xlabel('Predicted')
        th=cm.max()/2.
        for a in range(cm.shape[0]):
            for b in range(cm.shape[1]):
                axes[i].text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=8,fontweight='bold',color='white' if cm[a,b]>th else 'black')
    plt.tight_layout()
    plt.savefig('charts/chart2_confusion_matrices.png',dpi=300,bbox_inches='tight')
    plt.close()
    print("    SAVED: charts/chart2_confusion_matrices.png")

# CHART 3: Summary
print("  [3/3] Summary chart...")
fig,ax=plt.subplots(figsize=(10,5))
labels=[]; accs=[]; cols=['#3498db','#2ecc71','#e74c3c']
for k in R:
    d=R[k]; labels.append(f"{d['best_model']}\n({titles_map.get(k,k).replace(chr(10),' ')})"); accs.append(d['best_accuracy']*100)
bars=ax.barh(range(len(labels)),accs,color=cols[:len(labels)],edgecolor='black',height=0.5)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=10)
ax.set_xlabel('Accuracy (%)'); ax.set_title('Best Model Per Task — Employee Monitoring System',fontsize=13,fontweight='bold')
ax.set_xlim(0,110)
for b,v in zip(bars,accs):
    ax.text(v+0.5,b.get_y()+b.get_height()/2.,f'{v:.2f}%',va='center',fontsize=12,fontweight='bold')
plt.tight_layout()
plt.savefig('charts/chart3_best_models.png',dpi=300,bbox_inches='tight')
plt.close()
print("    SAVED: charts/chart3_best_models.png")

# ================================================================
#  GRAND SUMMARY
# ================================================================
total_time=time.time()-start_time
header(f"DONE! Total time: {total_time:.0f} seconds")

print("\n  RESULTS SUMMARY:")
for k,d in R.items():
    t=titles_map.get(k,k).replace('\n',' ')
    print(f"\n  {t}")
    print(f"    Dataset: {d['dataset']} ({d.get('samples','?')} samples)")
    print(f"    BEST: {d['best_model']} -> {d['best_accuracy']:.2%}")
    for m,s in d['scores'].items():
        mk=" <-- BEST" if m==d['best_model'] else ""
        print(f"      {m:>20}: {s:.4f} ({s:.2%}){mk}")

print(f"""
  FILES:
    models/real_data/kaggle_results.json
    charts/chart1_accuracy_comparison.png
    charts/chart2_confusion_matrices.png
    charts/chart3_best_models.png
""")
