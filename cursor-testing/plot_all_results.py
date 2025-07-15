import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

evaluation_files = {
    'Claude 4 Sonnet': 'data/cursor_evaluation/cursor.cursor_evaluation.json',
    'Auto': 'data/cursor_evaluation/cursor.auto_evaluation.json', 
    'Gemini 2.5 Pro': 'data/cursor_evaluation/cursor.gemini_evaluation.json',
    'GPT-4.1': 'data/cursor_evaluation/cursor.gpt_evaluation.json'
}

data = {}
for name, filename in evaluation_files.items():
    with open(filename, 'r') as f:
        data[name] = json.load(f)

models = list(data.keys())
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Prepare data for all visualizations
all_submitted = set(data['Claude 4 Sonnet']['submitted_ids'])
instance_list = sorted(list(all_submitted))
resolved_counts = [data[model]['resolved_instances'] for model in models]
completed_counts = [data[model]['completed_instances'] for model in models]
success_rates = [resolved/completed*100 if completed > 0 else 0 for resolved, completed in zip(resolved_counts, completed_counts)]

# Graph 1: Success Rate Comparison
fig1, ax1 = plt.subplots(figsize=(12, 8))
bars1 = ax1.bar(models, success_rates, color=colors)
ax1.set_title('Success Rate Comparison\n(Resolved/Completed %)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_ylim(0, max(success_rates) * 1.1 if success_rates else 100)

# Add value labels on bars
for bar, rate, resolved, completed in zip(bars1, success_rates, resolved_counts, completed_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%\n({resolved}/{completed})', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('docs/assets/success_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 2: Instance Completion Matrix (Heatmap)
fig2, ax2 = plt.subplots(figsize=(12, 16))

# Create completion matrix
completion_matrix = []
for instance in instance_list:
    row = []
    for model in models:
        resolved_ids = set(data[model]['resolved_ids'])
        row.append(1 if instance in resolved_ids else 0)
    completion_matrix.append(row)

# Create DataFrame for better visualization
df = pd.DataFrame(completion_matrix, index=instance_list, columns=models)

# Create heatmap
sns.heatmap(df, annot=True, cmap='RdYlGn', cbar=True, 
            xticklabels=True, yticklabels=True, ax=ax2, 
            square=False, linewidths=0.5)
ax2.set_title('Instance Completion Matrix\n(Green = Resolved, Red = Not Resolved)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Models', fontsize=12)
ax2.set_ylabel('Instances', fontsize=12)

# Rotate x-axis labels for better readability
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('docs/assets/instance_completion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 3: Performance Summary Table
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Create summary statistics
summary_data = []
for model in models:
    resolved = data[model]['resolved_instances']
    unresolved = data[model]['unresolved_instances']
    total = resolved + unresolved
    success_rate = (resolved / total * 100) if total > 0 else 0
    summary_data.append([model, resolved, unresolved, total, f"{success_rate:.1f}%"])

# Create table
table = ax3.table(cellText=summary_data,
                  colLabels=['Model', 'Resolved', 'Unresolved', 'Total', 'Success Rate'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Color code the success rate column
for i in range(len(summary_data)):
    success_rate = float(summary_data[i][4].rstrip('%'))
    if success_rate >= 60:
        table[(i+1, 4)].set_facecolor('#90EE90')  # Light green
    elif success_rate >= 40:
        table[(i+1, 4)].set_facecolor('#FFD700')  # Gold
    else:
        table[(i+1, 4)].set_facecolor('#FFB6C1')  # Light red

# Style header row
for j in range(5):
    table[(0, j)].set_facecolor('#4CAF50')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax3.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
ax3.axis('off')

plt.tight_layout()
plt.savefig('docs/assets/performance_summary_table.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 4: Detailed Instance Analysis
fig4, ax4 = plt.subplots(figsize=(14, 10))

# Create a stacked bar chart showing which models solved each instance
instance_counts = {}
for instance in instance_list:
    count = 0
    for model in models:
        if instance in set(data[model]['resolved_ids']):
            count += 1
    instance_counts[instance] = count

# Sort instances by how many models solved them
sorted_instances = sorted(instance_counts.items(), key=lambda x: x[1], reverse=True)
instance_names = [item[0] for item in sorted_instances]
solver_counts = [item[1] for item in sorted_instances]

# Create color mapping based on number of solvers
colors_map = ['#d32f2f', '#ff9800', '#ffc107', '#4caf50']  # Red to Green
bar_colors = [colors_map[count] if count < 4 else colors_map[3] for count in solver_counts]

bars = ax4.barh(range(len(instance_names)), solver_counts, color=bar_colors)
ax4.set_yticks(range(len(instance_names)))
ax4.set_yticklabels(instance_names, fontsize=8)
ax4.set_xlabel('Number of Models that Solved the Instance')
ax4.set_title('Instance Difficulty Analysis\n(Number of Models that Solved Each Instance)', fontsize=14, fontweight='bold')
ax4.set_xlim(0, 4)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, solver_counts)):
    width = bar.get_width()
    ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
             f'{count}', ha='left', va='center', fontweight='bold')

# Add legend

plt.tight_layout()
plt.savefig('docs/assets/instance_difficulty_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed analysis
print("=== Model Completion Analysis ===")
print(f"{'Model':<20} {'Resolved':<10} {'Unresolved':<12} {'Total':<8} {'Success Rate':<12}")
print("-" * 70)

for model in models:
    resolved = data[model]['resolved_instances']
    unresolved = data[model]['unresolved_instances']
    total = resolved + unresolved
    success_rate = (resolved / total * 100) if total > 0 else 0
    print(f"{model:<20} {resolved:<10} {unresolved:<12} {total:<8} {success_rate:.1f}%")

print("\n=== Instance-by-Instance Analysis ===")
resolved_by_model = {}
for model in models:
    resolved_by_model[model] = set(data[model]['resolved_ids'])

print("\nInstances resolved by each model:")
for instance in sorted(all_submitted):
    resolvers = [model for model in models if instance in resolved_by_model[model]]
    if resolvers:
        print(f"  {instance}: {', '.join(resolvers)}")
    else:
        print(f"  {instance}: None")

# Find best and worst performing instances
print(f"\nInstances resolved by all models:")
all_resolved = set.intersection(*[resolved_by_model[model] for model in models])
for instance in sorted(all_resolved):
    print(f"  - {instance}")

print(f"\nInstances resolved by no model:")
none_resolved = all_submitted - set.union(*[resolved_by_model[model] for model in models])
for instance in sorted(none_resolved):
    print(f"  - {instance}")

# Model-specific analysis
print(f"\n=== Model-Specific Analysis ===")
for model in models:
    resolved_ids = resolved_by_model[model]
    print(f"\n{model} resolved instances:")
    for instance in sorted(resolved_ids):
        print(f"  - {instance}") 