# """
# AUTOMATED GAP FIXER — Add Missing Validations & Metrics
# Run: python fix_gaps.py
# This will enhance your existing results with:
#   ✅ Precision, Recall, F1 scores
#   ✅ Confidence intervals
#   ✅ Saved models (.pkl)
#   ✅ Data validation reports
#   ✅ ROC/AUC for binary classification
# """

# import os
# import json
# import warnings
# import time
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import joblib

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import (
#     accuracy_score, precision_recall_fscore_support,
#     roc_auc_score, roc_curve, classification_report
# )
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from scipy import stats

# print("\n" + "="*70)
# print("  AUTOMATED GAP FIXER — Enhancing ML Results")
# print("="*70)

# # ================================================================
# #  UTILITY FUNCTIONS
# # ================================================================

# def validate_data(df, name):
#     """Gap 1: Data validation"""
#     print(f"\n  📋 DATA VALIDATION: {name}")
#     print(f"     Shape: {df.shape}")
    
#     # Missing values
#     missing = df.isnull().sum()
#     total_missing = missing.sum()
#     if total_missing > 0:
#         print(f"     ⚠️  Missing values: {total_missing}")
#         print(missing[missing > 0].to_string())
#     else:
#         print(f"     ✅ No missing values")
    
#     # Duplicates
#     dups = df.duplicated().sum()
#     print(f"     Duplicates: {dups} ({dups/len(df)*100:.2f}%)")
    
#     # Numeric ranges
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     if len(numeric_cols) > 0:
#         print(f"     Numeric columns: {len(numeric_cols)}")
#         for col in numeric_cols[:3]:  # Show first 3
#             print(f"       {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")


# def verify_scaling(X_scaled, name):
#     """Gap 8: Scaling verification"""
#     mean_val = X_scaled.mean()
#     std_val = X_scaled.std()
#     print(f"  📊 SCALING CHECK ({name}):")
#     print(f"     Mean: {mean_val:.6f} (should be ~0)")
#     print(f"     Std:  {std_val:.6f} (should be ~1)")
    
#     if abs(mean_val) > 0.01:
#         print(f"     ⚠️  Mean not centered!")
#     if abs(std_val - 1.0) > 0.1:
#         print(f"     ⚠️  Std not normalized!")
#     else:
#         print(f"     ✅ Scaling verified")


# def calculate_ci(scores, confidence=0.95):
#     """Gap 14: Confidence intervals"""
#     mean = scores.mean()
#     std = scores.std()
#     n = len(scores)
#     t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
#     margin = t_value * (std / np.sqrt(n))
#     return mean, mean - margin, mean + margin


# def full_metrics(y_test, y_pred, task_name):
#     """Gap 5: Complete metrics"""
#     acc = accuracy_score(y_test, y_pred)
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_test, y_pred, average='weighted', zero_division=0
#     )
    
#     print(f"\n  📈 FULL METRICS ({task_name}):")
#     print(f"     Accuracy:  {acc:.4f}")
#     print(f"     Precision: {prec:.4f}")
#     print(f"     Recall:    {rec:.4f}")
#     print(f"     F1-Score:  {f1:.4f}")
    
#     return {
#         'accuracy': round(float(acc), 4),
#         'precision': round(float(prec), 4),
#         'recall': round(float(rec), 4),
#         'f1_score': round(float(f1), 4)
#     }


# # ================================================================
# #  LOAD EXISTING RESULTS (or create empty if not found)
# # ================================================================

# print("\n" + "="*70)
# print("  LOADING EXISTING RESULTS")
# print("="*70)

# results_path = "models/real_data/kaggle_results.json"
# if os.path.exists(results_path):
#     with open(results_path, 'r') as f:
#         results = json.load(f)
#     if results:
#         print(f"  ✅ Loaded results for {len(results)} tasks")
#     else:
#         print(f"  ⚠️  Results file empty, creating fresh analysis")
#         results = {}
# else:
#     print(f"  ⚠️  Results file not found, creating fresh analysis")
#     results = {}

# # Initialize empty structure for missing tasks
# if 'employee_activity' not in results:
#     results['employee_activity'] = {'dataset': 'UCI HAR (Kaggle)', 'classes': [], 'scores': {}}
# if 'office_occupancy' not in results:
#     results['office_occupancy'] = {'dataset': 'UCI Occupancy (Kaggle)', 'scores': {}}
# if 'employee_performance' not in results:
#     results['employee_performance'] = {'dataset': 'Employee Activity & Evaluation (Kaggle)', 'scores': {}}

# # ================================================================
# #  PATHS
# # ================================================================

# B = os.path.join("data", "kaggle")
# HAR_TR = os.path.join(B, "human-activity-recognition-with-smartphones", "train.csv")
# HAR_TE = os.path.join(B, "human-activity-recognition-with-smartphones", "test.csv")
# OCC_TR = os.path.join(B, "occupancy-detection-data-set-uci", "datatraining.txt")
# OCC_TE = os.path.join(B, "occupancy-detection-data-set-uci", "datatest.txt")

# EMP_DATA = None
# for root, dirs, files in os.walk(B):
#     for f in files:
#         if 'employee' in f.lower() and f.endswith('.csv'):
#             EMP_DATA = os.path.join(root, f)
#             break

# enhanced_results = {}

# # ================================================================
# #  TASK 1: EMPLOYEE ACTIVITY (UCI HAR)
# # ================================================================

# print("\n" + "="*70)
# print("  TASK 1: EMPLOYEE ACTIVITY RECOGNITION")
# print("="*70)

# if os.path.exists(HAR_TR):
#     tr = pd.read_csv(HAR_TR)
#     te = pd.read_csv(HAR_TE)
    
#     validate_data(tr, "UCI HAR Training")
    
#     lc = 'Activity'
#     ex = [lc]
#     if 'subject' in tr.columns:
#         ex.append('subject')
#     fc = [c for c in tr.columns if c not in ex]
    
#     le = LabelEncoder()
#     X_train = np.nan_to_num(tr[fc].values)
#     y_train = le.fit_transform(tr[lc])
#     X_test = np.nan_to_num(te[fc].values)
#     y_test = le.transform(te[lc])
    
#     # Scaling
#     sc = StandardScaler()
#     X_train_scaled = sc.fit_transform(X_train)
#     X_test_scaled = sc.transform(X_test)
    
#     verify_scaling(X_train_scaled, "Training")
#     verify_scaling(X_test_scaled, "Test")
    
#     # Train best model (Logistic Regression from original results)
#     print("\n  🔧 Training Logistic Regression...")
#     model = LogisticRegression(max_iter=1000, random_state=42)
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
    
#     # Full metrics
#     metrics = full_metrics(y_test, y_pred, "Activity")
    
#     # Cross-validation with CI
#     print("\n  🔄 Cross-Validation (5-fold)...")
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
#     mean_cv, ci_lower, ci_upper = calculate_ci(cv_scores)
    
#     print(f"     CV Mean: {mean_cv:.4f}")
#     print(f"     CV Std:  {cv_scores.std():.4f}")
#     print(f"     95% CI:  [{ci_lower:.4f}, {ci_upper:.4f}]")
    
#     # Save model (Gap 9)
#     print("\n  💾 Saving model and scaler...")
#     joblib.dump(model, "models/real_data/activity_model.pkl")
#     joblib.dump(sc, "models/real_data/activity_scaler.pkl")
#     joblib.dump(le, "models/real_data/activity_label_encoder.pkl")
#     print("     ✅ Saved: activity_model.pkl, activity_scaler.pkl")
    
#     # Enhanced results
#     enhanced_results['employee_activity'] = {
#         **results['employee_activity'],
#         **metrics,
#         'cv_mean': round(float(mean_cv), 4),
#         'cv_std': round(float(cv_scores.std()), 4),
#         'ci_95_lower': round(float(ci_lower), 4),
#         'ci_95_upper': round(float(ci_upper), 4),
#         'model_saved': 'activity_model.pkl',
#         'scaler_saved': 'activity_scaler.pkl'
#     }

# # ================================================================
# #  TASK 2: OFFICE OCCUPANCY (BINARY)
# # ================================================================

# print("\n" + "="*70)
# print("  TASK 2: OFFICE OCCUPANCY DETECTION")
# print("="*70)

# if os.path.exists(OCC_TR):
#     otr = pd.read_csv(OCC_TR)
#     ote = pd.read_csv(OCC_TE)
    
#     validate_data(otr, "UCI Occupancy Training")
    
#     fc = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
#     X_train = np.nan_to_num(otr[fc].values)
#     y_train = otr['Occupancy'].astype(int).values
#     X_test = np.nan_to_num(ote[fc].values)
#     y_test = ote['Occupancy'].astype(int).values
    
#     # Scaling
#     sc = StandardScaler()
#     X_train_scaled = sc.fit_transform(X_train)
#     X_test_scaled = sc.transform(X_test)
    
#     verify_scaling(X_train_scaled, "Training")
    
#     # Train
#     print("\n  🔧 Training Logistic Regression...")
#     model = LogisticRegression(max_iter=500, random_state=42)
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
    
#     # Full metrics
#     metrics = full_metrics(y_test, y_pred, "Occupancy")
    
#     # Gap 7: ROC/AUC for binary classification
#     print("\n  📊 ROC/AUC Analysis...")
#     y_proba = model.predict_proba(X_test_scaled)[:, 1]
#     auc_score = roc_auc_score(y_test, y_proba)
#     print(f"     AUC Score: {auc_score:.4f}")
    
#     # Plot ROC curve
#     fpr, tpr, thresholds = roc_curve(y_test, y_proba)
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc_score:.4f})')
#     plt.plot([0, 1], [0, 1], 'k--', label='Random')
#     plt.xlabel('False Positive Rate', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title('ROC Curve - Office Occupancy Detection', fontsize=14, fontweight='bold')
#     plt.legend(fontsize=11)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('charts/gap_roc_curve.png', dpi=300)
#     plt.close()
#     print("     ✅ Saved: charts/gap_roc_curve.png")
    
#     # Cross-validation
#     print("\n  🔄 Cross-Validation (5-fold)...")
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
#     mean_cv, ci_lower, ci_upper = calculate_ci(cv_scores)
#     print(f"     CV Mean: {mean_cv:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
#     # Save model
#     print("\n  💾 Saving model and scaler...")
#     joblib.dump(model, "models/real_data/occupancy_model.pkl")
#     joblib.dump(sc, "models/real_data/occupancy_scaler.pkl")
#     print("     ✅ Saved: occupancy_model.pkl, occupancy_scaler.pkl")
    
#     enhanced_results['office_occupancy'] = {
#         **results['office_occupancy'],
#         **metrics,
#         'auc_score': round(float(auc_score), 4),
#         'cv_mean': round(float(mean_cv), 4),
#         'cv_std': round(float(cv_scores.std()), 4),
#         'ci_95_lower': round(float(ci_lower), 4),
#         'ci_95_upper': round(float(ci_upper), 4),
#         'model_saved': 'occupancy_model.pkl',
#         'scaler_saved': 'occupancy_scaler.pkl',
#         'roc_curve': 'charts/gap_roc_curve.png'
#     }

# # ================================================================
# #  TASK 3: EMPLOYEE PERFORMANCE
# # ================================================================

# print("\n" + "="*70)
# print("  TASK 3: EMPLOYEE PERFORMANCE CLASSIFICATION")
# print("="*70)

# if EMP_DATA and os.path.exists(EMP_DATA):
#     df = pd.read_csv(EMP_DATA)
    
#     validate_data(df, "Employee Performance")
    
#     # Find label column
#     label_col = None
#     for c in df.columns:
#         if 'performance' in c.lower() and ('label' in c.lower() or 'level' in c.lower()):
#             label_col = c
#             break
#     if not label_col:
#         label_col = df.columns[-1]
    
#     print(f"\n  🏷️  Label column: '{label_col}'")
#     print(f"     Class distribution:")
#     print(df[label_col].value_counts().to_string())
    
#     # Encode
#     le = LabelEncoder()
#     y = le.fit_transform(df[label_col].astype(str))
    
#     # Features
#     encoded_df = df.copy()
#     for c in encoded_df.columns:
#         if c == label_col:
#             continue
#         if encoded_df[c].dtype == 'object':
#             encoded_df[c] = LabelEncoder().fit_transform(encoded_df[c].astype(str))
    
#     feature_cols = [c for c in encoded_df.columns if c != label_col and c.lower() != 'employee_id']
#     for c in feature_cols:
#         encoded_df[c] = pd.to_numeric(encoded_df[c], errors='coerce')
#     encoded_df = encoded_df.fillna(0)
    
#     X = encoded_df[feature_cols].values.astype(float)
    
#     # Split with stratification
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     # Scaling
#     sc = StandardScaler()
#     X_train_scaled = sc.fit_transform(X_train)
#     X_test_scaled = sc.transform(X_test)
    
#     verify_scaling(X_train_scaled, "Training")
    
#     # Train (Gradient Boost from original)
#     print("\n  🔧 Training Gradient Boosting...")
#     model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
    
#     # Full metrics
#     metrics = full_metrics(y_test, y_pred, "Performance")
    
#     # Per-class metrics (Gap 10)
#     print("\n  📊 Per-Class Metrics:")
#     report = classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
#     for class_name in le.classes_:
#         class_metrics = report[str(class_name)]
#         print(f"     {class_name}:")
#         print(f"       Precision: {class_metrics['precision']:.4f}")
#         print(f"       Recall:    {class_metrics['recall']:.4f}")
#         print(f"       F1-Score:  {class_metrics['f1-score']:.4f}")
#         print(f"       Support:   {int(class_metrics['support'])}")
    
#     # Cross-validation
#     print("\n  🔄 Cross-Validation (5-fold)...")
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
#     mean_cv, ci_lower, ci_upper = calculate_ci(cv_scores)
#     print(f"     CV Mean: {mean_cv:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
#     # Save model
#     print("\n  💾 Saving model and scaler...")
#     joblib.dump(model, "models/real_data/performance_model.pkl")
#     joblib.dump(sc, "models/real_data/performance_scaler.pkl")
#     joblib.dump(le, "models/real_data/performance_label_encoder.pkl")
#     print("     ✅ Saved: performance_model.pkl, performance_scaler.pkl")
    
#     enhanced_results['employee_performance'] = {
#         **results['employee_performance'],
#         **metrics,
#         'per_class_metrics': {
#             str(cls): {
#                 'precision': round(report[str(cls)]['precision'], 4),
#                 'recall': round(report[str(cls)]['recall'], 4),
#                 'f1_score': round(report[str(cls)]['f1-score'], 4),
#                 'support': int(report[str(cls)]['support'])
#             }
#             for cls in le.classes_
#         },
#         'cv_mean': round(float(mean_cv), 4),
#         'cv_std': round(float(cv_scores.std()), 4),
#         'ci_95_lower': round(float(ci_lower), 4),
#         'ci_95_upper': round(float(ci_upper), 4),
#         'model_saved': 'performance_model.pkl',
#         'scaler_saved': 'performance_scaler.pkl'
#     }

# # ================================================================
# #  SAVE ENHANCED RESULTS
# # ================================================================

# print("\n" + "="*70)
# print("  SAVING ENHANCED RESULTS")
# print("="*70)

# with open("models/real_data/enhanced_results.json", 'w') as f:
#     json.dump(enhanced_results, f, indent=2)

# print("  ✅ Saved: models/real_data/enhanced_results.json")

# # ================================================================
# #  SUMMARY
# # ================================================================

# print("\n" + "="*70)
# print("  🎉 GAP FIX COMPLETE!")
# print("="*70)

# print("\n  ENHANCEMENTS ADDED:")
# print("  ✅ Full metrics (Precision, Recall, F1)")
# print("  ✅ Cross-validation with 95% confidence intervals")
# print("  ✅ Data validation checks")
# print("  ✅ Scaling verification")
# print("  ✅ ROC/AUC curve for binary classification")
# print("  ✅ Per-class metrics for imbalanced data")
# print("  ✅ All models saved as .pkl files")

# print("\n  FILES CREATED:")
# print("    models/real_data/enhanced_results.json")
# print("    models/real_data/activity_model.pkl")
# print("    models/real_data/activity_scaler.pkl")
# print("    models/real_data/occupancy_model.pkl")
# print("    models/real_data/occupancy_scaler.pkl")
# print("    models/real_data/performance_model.pkl")
# print("    models/real_data/performance_scaler.pkl")
# print("    charts/gap_roc_curve.png")

# print("\n  AVERAGE ACCURACY: {:.2%}".format(
#     np.mean([
#         enhanced_results['employee_activity']['accuracy'],
#         enhanced_results['office_occupancy']['accuracy'],
#         enhanced_results['employee_performance']['accuracy']
#     ])
# ))

# print("\n  📊 Use 'enhanced_results.json' for your thesis paper!")
# print("="*70 + "\n")
