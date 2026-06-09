import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("charts", exist_ok=True)

print("="*50)
print("  GENERATING CHARTS FROM SAVED RESULTS")
print("  (No re-training - using JSON results)")
print("="*50)

with open("models/real_data/kaggle_results.json", 'r') as f:
    results = json.load(f)

print(f"\n  Loaded results for: {list(results.keys())}")


print("\n[1/4] Chart 1: Accuracy Comparison Bar Chart...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

tasks = [
    ('activity_recognition', 'Activity Recognition\n(UCI HAR - 10,299 samples)', '#3498db', (80, 100)),
    ('occupancy_detection', 'Occupancy Detection\n(PIR Sensor - 10,808 samples)', '#e74c3c', (80, 100)),
    ('student_behaviour', 'Student Behaviour\n(xAPI-Edu - 480 samples)', '#9b59b6', (40, 80)),
]

for idx, (key, title, color, ylim) in enumerate(tasks):
    if key not in results:
        continue

    data = results[key]
    names = list(data['scores'].keys())
    scores = [v * 100 for v in data['scores'].values()]
    best_score = data['best_accuracy'] * 100

    colors = ['#2ecc71' if s == max(scores) else color for s in scores]

    bars = axes[idx].bar(range(len(names)), scores, color=colors, edgecolor='black', linewidth=0.5)
    axes[idx].set_xticks(range(len(names)))
    axes[idx].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Accuracy (%)', fontsize=12)
    axes[idx].set_ylim(ylim)
    axes[idx].axhline(y=best_score, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_score:.2f}%')
    axes[idx].legend(fontsize=9)

    for bar, val in zip(bars, scores):
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/chart1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  SAVED: charts/chart1_accuracy_comparison.png")


print("[2/4] Chart 2: Best Model Summary...")

fig, ax = plt.subplots(figsize=(10, 6))

task_labels = []
best_accs = []
bar_colors = ['#3498db', '#2ecc71', '#e74c3c']

for key in ['activity_recognition', 'occupancy_detection', 'student_behaviour']:
    if key in results:
        d = results[key]
        label = f"{d['best_model']}\n({key.replace('_',' ').title()})"
        task_labels.append(label)
        best_accs.append(d['best_accuracy'] * 100)

bars = ax.barh(range(len(task_labels)), best_accs, color=bar_colors[:len(task_labels)],
               edgecolor='black', linewidth=0.5, height=0.5)
ax.set_yticks(range(len(task_labels)))
ax.set_yticklabels(task_labels, fontsize=11)
ax.set_xlabel('Accuracy (%)', fontsize=13)
ax.set_title('Best Model Performance Across All Tasks\n(Real Kaggle Data)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)

for bar, val in zip(bars, best_accs):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2.,
            f'{val:.2f}%', ha='left', va='center', fontsize=13, fontweight='bold')

ax.axvline(x=90, color='gray', linestyle=':', alpha=0.3)
ax.text(90.5, -0.3, '90% threshold', color='gray', fontsize=9)

plt.tight_layout()
plt.savefig('charts/chart2_best_models_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  SAVED: charts/chart2_best_models_summary.png")


print("[3/4] Chart 3: All Models Heatmap...")

fig, ax = plt.subplots(figsize=(12, 5))

all_model_names = set()
for key in results:
    all_model_names.update(results[key]['scores'].keys())
all_model_names = sorted(all_model_names)

task_names = []
score_matrix = []

for key in ['activity_recognition', 'occupancy_detection', 'student_behaviour']:
    if key in results:
        task_names.append(key.replace('_', ' ').title())
        row = []
        for model in all_model_names:
            if model in results[key]['scores']:
                row.append(results[key]['scores'][model] * 100)
            else:
                row.append(0)
        score_matrix.append(row)

score_array = np.array(score_matrix)

im = ax.imshow(score_array, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
ax.set_xticks(range(len(all_model_names)))
ax.set_xticklabels(all_model_names, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(task_names)))
ax.set_yticklabels(task_names, fontsize=11)
ax.set_title('Model Accuracy Heatmap — All Tasks & Algorithms\n(Real Kaggle Data)', fontsize=14, fontweight='bold')

for i in range(len(task_names)):
    for j in range(len(all_model_names)):
        val = score_array[i, j]
        if val > 0:
            color = 'white' if val > 75 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

fig.colorbar(im, ax=ax, label='Accuracy (%)')
plt.tight_layout()
plt.savefig('charts/chart3_heatmap_all_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("  SAVED: charts/chart3_heatmap_all_models.png")


print("[4/4] Chart 4: Dataset Summary...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

datasets = ['UCI HAR\n(Activity)', 'UCI Occupancy\n(PIR Sensor)', 'xAPI-Edu\n(Behaviour)']
sizes = []
for key in ['activity_recognition', 'occupancy_detection', 'student_behaviour']:
    if key in results:
        d = results[key]
        total = d.get('train_samples', 0) + d.get('test_samples', 0)
        if total == 0:
            total = d.get('samples', 0)
        sizes.append(total)

bars = axes[0].bar(datasets, sizes, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.5)
axes[0].set_title('Dataset Sizes', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Samples', fontsize=12)
for bar, val in zip(bars, sizes):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

task_short = ['Activity\nRecognition', 'Occupancy\nDetection', 'Student\nBehaviour']
best_vals = [results[k]['best_accuracy'] * 100 for k in ['activity_recognition', 'occupancy_detection', 'student_behaviour'] if k in results]
bar_colors = ['#3498db', '#2ecc71', '#e74c3c']

bars2 = axes[1].bar(task_short[:len(best_vals)], best_vals, color=bar_colors[:len(best_vals)], edgecolor='black', linewidth=0.5)
axes[1].set_title('Best Accuracy Per Task', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_ylim(0, 110)
for bar, val in zip(bars2, best_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/chart4_dataset_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  SAVED: charts/chart4_dataset_summary.png")


print("\n" + "="*50)
print("  ALL CHARTS DONE! (from JSON - no re-training)")
print("="*50)
print("""
  charts/chart1_accuracy_comparison.png   - Bar charts (3 tasks)
  charts/chart2_best_models_summary.png   - Horizontal bar (best models)
  charts/chart3_heatmap_all_models.png    - Heatmap (all scores)
  charts/chart4_dataset_summary.png       - Dataset sizes + best accuracy

  -> Open charts/ folder -> Insert into Word/Paper!
""")
