"""
THESIS MASTER SCRIPT - WINDOWS COMPATIBLE (NO EMOJI)
Run: python run_all_FIXED.py
Runs all 3 necessary scripts without deleting previous work
"""

import os
import sys
import time
import subprocess

print("\n" + "="*70)
print("THESIS MASTER SCRIPT - COMPLETE ML PIPELINE")
print("="*70)
print("\nThis will run 3 scripts in sequence:")
print("  [1/3] train_with_real_data.py    (~70 seconds)")
print("  [2/3] deep_analysis.py            (~140 seconds)")
print("  [3/3] fix_gaps.py                 (~30 seconds)")
print("\nEstimated total time: 4-5 minutes")
print("="*70)
print("\nStarting in 3 seconds...")
time.sleep(3)

START_TIME = time.time()
RESULTS = {}

def run_script(script_name, step_num, total_steps):
    """Run a Python script and capture results."""
    print("\n" + "="*70)
    print(f"[{step_num}/{total_steps}] RUNNING: {script_name}")
    print("="*70)
    
    start = time.time()
    
    try:
        # Set UTF-8 encoding for subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            encoding='utf-8',
            errors='replace'  # Replace problematic characters
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"\n[SUCCESS] ({elapsed:.1f}s)")
            RESULTS[script_name] = {'status': 'success', 'time': elapsed}
            return True
        else:
            print(result.stdout)
            print(result.stderr)
            print(f"\n[FAILED] ({elapsed:.1f}s)")
            RESULTS[script_name] = {'status': 'failed', 'time': elapsed}
            return False
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n[TIMEOUT] ({elapsed:.1f}s)")
        RESULTS[script_name] = {'status': 'timeout', 'time': elapsed}
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[ERROR] {e} ({elapsed:.1f}s)")
        RESULTS[script_name] = {'status': 'error', 'time': elapsed, 'error': str(e)}
        return False


# Run scripts (skip train_with_validation.py as it deletes work)
scripts = [
    "train_with_real_data.py",
    "deep_analysis.py",
    "fix_gaps.py"
]

all_success = True

for i, script in enumerate(scripts, 1):
    success = run_script(script, i, len(scripts))
    if not success:
        all_success = False
        print(f"\n[WARNING] {script} failed, but continuing...")

TOTAL_TIME = time.time() - START_TIME

# Summary
print("\n" + "="*70)
print("THESIS MASTER SCRIPT COMPLETE")
print("="*70)

print("\nEXECUTION SUMMARY:")
for script, result in RESULTS.items():
    status_icon = "[OK]" if result['status'] == 'success' else "[FAIL]"
    print(f"  {status_icon} {script:30} {result['time']:>6.1f}s  {result['status']}")

print(f"\nTOTAL TIME: {TOTAL_TIME:.1f}s ({TOTAL_TIME/60:.1f} minutes)")


# Check files
print("\n" + "="*70)
print("CHECKING OUTPUT FILES")
print("="*70)

expected_files = [
    "models/real_data/kaggle_results.json",
    "models/real_data/enhanced_results.json",
    "models/real_data/deep_analysis.json",
    "models/real_data/activity_model.pkl",
    "models/real_data/occupancy_model.pkl",
    "models/real_data/performance_model.pkl",
    "charts/chart1_accuracy_comparison.png",
    "charts/chart2_confusion_matrices.png",
    "charts/chart3_best_models.png",
    "charts/deep3_cross_validation.png",
    "charts/gap_roc_curve.png",
]

found = 0
for file_path in expected_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024
        print(f"  [OK] {file_path} ({size:.1f} KB)")
        found += 1
    else:
        print(f"  [MISSING] {file_path}")

print(f"\nFILES FOUND: {found}/{len(expected_files)}")

# Show final stats
if os.path.exists("models/real_data/enhanced_results.json"):
    import json
    
    print("\n" + "="*70)
    print("FINAL ML RESULTS")
    print("="*70)
    
    with open("models/real_data/enhanced_results.json", 'r') as f:
        results = json.load(f)
    
    accuracies = []
    
    for task, data in results.items():
        task_name = {
            'employee_activity': 'Activity Recognition',
            'office_occupancy': 'Occupancy Detection',
            'employee_performance': 'Performance Classification'
        }.get(task, task)
        
        acc = data.get('accuracy', data.get('best_accuracy', 0))
        cv = data.get('cv_mean', 0)
        
        accuracies.append(acc)
        
        print(f"\n{task_name}:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  CV Mean:  {cv:.4f} ({cv*100:.2f}%)")
    
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
    print(f"\nAVERAGE ACCURACY: {avg_acc:.4f} ({avg_acc*100:.2f}%)")

print("\n" + "="*70)
if all_success and found >= 8:
    print("ALL WORK COMPLETE! YOUR THESIS IS READY!")
else:
    print("PARTIAL SUCCESS - Some files may be missing")
print("="*70 + "\n")
