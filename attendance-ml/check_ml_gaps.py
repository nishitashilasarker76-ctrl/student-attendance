"""
ML WORK VALIDATION CHECKER
===========================
This script checks for common ML gaps in your thesis work
Run: python check_ml_gaps.py
"""

import os
import json
import numpy as np

print("\n" + "="*80)
print("ML WORK GAP CHECKER - THESIS VALIDATION")
print("="*80)

ISSUES = []
WARNINGS = []
PASS = []

def check(name, condition, error_msg, warning_msg=None):
    """Check a condition and categorize result"""
    if condition:
        PASS.append(f"[OK] {name}")
    elif warning_msg:
        WARNINGS.append(f"[WARNING] {name}: {warning_msg}")
    else:
        ISSUES.append(f"[ISSUE] {name}: {error_msg}")

print("\n[1/10] Checking if results file exists...")
results_exists = os.path.exists("models/real_data/thesis_results.json")
check("Results File", results_exists, "thesis_results.json not found - run thesis_complete.py")

if results_exists:
    with open("models/real_data/thesis_results.json", 'r') as f:
        results = json.load(f)
    
    print("[2/10] Checking number of tasks...")
    check("Task Count", len(results) == 3, 
          f"Expected 3 tasks, found {len(results)}")
    
    print("[3/10] Checking if all metrics present...")
    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for task, data in results.items():
        for metric in required_metrics:
            check(f"{task} - {metric}", 
                  metric in data.get('metrics', {}),
                  f"Missing {metric} in {task}")
    
    print("[4/10] Checking cross-validation...")
    for task, data in results.items():
        has_cv = 'cv_results' in data and len(data['cv_results']) > 0
        check(f"{task} - Cross Validation", has_cv,
              f"No cross-validation in {task}")
        
        if has_cv:
            for model, cv_data in data['cv_results'].items():
                has_cv_mean = 'cv_mean' in cv_data
                has_cv_std = 'cv_std' in cv_data
                check(f"{task} - {model} CV stats", 
                      has_cv_mean and has_cv_std,
                      f"Missing CV mean/std for {model}")
    
    print("[5/10] Checking accuracy levels...")
    for task, data in results.items():
        acc = data.get('metrics', {}).get('accuracy', 0)
        check(f"{task} - Accuracy", acc >= 0.85,
              f"Low accuracy: {acc:.2%}",
              f"Accuracy {acc:.2%} could be better (aim for >90%)" if acc < 0.90 else None)
    
    print("[6/10] Checking confusion matrices...")
    for task, data in results.items():
        has_cm = 'confusion_matrix' in data and len(data['confusion_matrix']) > 0
        check(f"{task} - Confusion Matrix", has_cm,
              f"Missing confusion matrix in {task}")
    
    print("[7/10] Checking for class imbalance handling...")
    for task, data in results.items():
        if task == 'performance':
            # Check if class distribution exists
            classes = data.get('classes', [])
            check(f"{task} - Class info", len(classes) > 0,
                  "No class information found")
    
    print("[8/10] Checking model persistence...")
    model_files = [
        "models/real_data/activity_model.pkl",
        "models/real_data/occupancy_model.pkl",
        "models/real_data/performance_model.pkl"
    ]
    for model_file in model_files:
        check(f"Model file - {model_file.split('/')[-1]}", 
              os.path.exists(model_file),
              f"{model_file} not found")
    
    scaler_files = [
        "models/real_data/activity_scaler.pkl",
        "models/real_data/occupancy_scaler.pkl",
        "models/real_data/performance_scaler.pkl"
    ]
    for scaler_file in scaler_files:
        check(f"Scaler file - {scaler_file.split('/')[-1]}", 
              os.path.exists(scaler_file),
              f"{scaler_file} not found")
    
    print("[9/10] Checking charts...")
    required_charts = [
        "charts/accuracy_comparison.png",
        "charts/confusion_matrices.png",
        "charts/best_models_summary.png",
        "charts/roc_curve.png",
        "charts/cross_validation.png",
        "charts/detailed_metrics.png"
    ]
    for chart in required_charts:
        check(f"Chart - {chart.split('/')[-1]}", 
              os.path.exists(chart),
              f"{chart} not found")
    
    print("[10/10] Checking for statistical rigor...")
    
    # Check CV std is reasonable (not too high)
    for task, data in results.items():
        if 'cv_results' in data and data.get('best_model'):
            best_model = data['best_model']
            cv_std = data['cv_results'][best_model].get('cv_std', 0)
            check(f"{task} - CV stability", cv_std < 0.05,
                  f"High CV std ({cv_std:.4f}) indicates unstable model",
                  f"CV std {cv_std:.4f} is acceptable but could be better" if cv_std < 0.10 else None)
    
    # Check for overfitting
    for task, data in results.items():
        test_acc = data.get('metrics', {}).get('accuracy', 0)
        if 'cv_results' in data and data.get('best_model'):
            cv_mean = data['cv_results'][data['best_model']].get('cv_mean', 0)
            gap = abs(test_acc - cv_mean)
            check(f"{task} - No overfitting", gap < 0.05,
                  f"Large gap ({gap:.2%}) between test and CV suggests overfitting",
                  f"Gap {gap:.2%} is acceptable" if gap < 0.10 else None)

else:
    print("[SKIPPED] Cannot check details without results file")

# ================================================================
#  REPORT
# ================================================================

print("\n" + "="*80)
print("VALIDATION REPORT")
print("="*80)

print(f"\nPassed: {len(PASS)}")
print(f"Warnings: {len(WARNINGS)}")
print(f"Issues: {len(ISSUES)}")

if ISSUES:
    print("\n" + "="*80)
    print("CRITICAL ISSUES (Must Fix):")
    print("="*80)
    for issue in ISSUES:
        print(f"  {issue}")

if WARNINGS:
    print("\n" + "="*80)
    print("WARNINGS (Recommended to fix):")
    print("="*80)
    for warning in WARNINGS:
        print(f"  {warning}")

if len(PASS) > 0:
    print("\n" + "="*80)
    print(f"PASSED CHECKS: {len(PASS)}")
    print("="*80)

# ================================================================
#  ADDITIONAL RECOMMENDATIONS
# ================================================================

print("\n" + "="*80)
print("ADDITIONAL RECOMMENDATIONS FOR THESIS DEFENSE:")
print("="*80)

recommendations = []

if results_exists:
    # Check for confidence intervals
    has_ci = False
    for task, data in results.items():
        if 'cv_results' in data:
            for model, cv_data in data['cv_results'].items():
                if 'ci_lower' in cv_data or 'ci_upper' in cv_data:
                    has_ci = True
    
    if not has_ci:
        recommendations.append(
            "Add 95% Confidence Intervals: Panel may ask about statistical significance"
        )
    
    # Check for AUC in binary classification
    has_auc = False
    for task, data in results.items():
        if len(data.get('classes', [])) == 2:
            if 'auc' in data.get('metrics', {}):
                has_auc = True
    
    if not has_auc and any(len(results[t].get('classes', [])) == 2 for t in results):
        recommendations.append(
            "Add ROC/AUC for binary task: Standard metric for binary classification"
        )
    
    # Check average accuracy
    avg_acc = np.mean([results[t]['metrics']['accuracy'] for t in results])
    if avg_acc < 0.90:
        recommendations.append(
            f"Consider hyperparameter tuning: Average accuracy {avg_acc:.2%} could be improved"
        )

if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("  None - Your work looks complete!")

# ================================================================
#  FINAL VERDICT
# ================================================================

print("\n" + "="*80)
if len(ISSUES) == 0 and len(WARNINGS) == 0:
    print("VERDICT: EXCELLENT - NO GAPS FOUND!")
    print("Your ML work is thesis-defense ready!")
elif len(ISSUES) == 0:
    print("VERDICT: GOOD - Minor warnings only")
    print("Your work is ready, but consider addressing warnings")
else:
    print("VERDICT: NEEDS WORK - Critical issues found")
    print("Please fix the issues above before defense")
print("="*80 + "\n")
