import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

# Data
question_types = ['Yes/No', 'Number', 'Other']
models = ['CLIP', 'BLIP-2', 'LLaVA', 'GPT-4V']
colors = ['#5E81AC', '#81A1C1', '#88C0D0', '#8FBCBB']

# Accuracy data
data = {
    'CLIP': [82.1, 43.2, 51.4],
    'BLIP-2': [87.4, 51.7, 58.2],
    'LLaVA': [89.2, 56.3, 63.1],
    'GPT-4V': [92.3, 63.8, 70.5]
}

# Convert to array
x = np.arange(len(question_types))
width = 0.2

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
for i, model in enumerate(models):
    ax.bar(x + i * width, data[model], width, label=model, color=colors[i])

# Customize x-axis
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(question_types, fontsize=14, fontweight='bold')

# Set y-axis limits and ticks
ax.set_ylim(40, 95)
ax.set_yticks([40, 50, 60, 70, 80, 90])
ax.set_yticklabels([str(int(t)) for t in ax.get_yticks()], fontsize=12)

# Add value labels on bars
for i, model in enumerate(models):
    for j, val in enumerate(data[model]):
        ax.text(x[j] + i * width, val + 1, f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=12, frameon=False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set y-axis label
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('vqa_accuracy_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved to vqa_accuracy_chart.png")