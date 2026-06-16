import matplotlib.pyplot as plt

# Extracted data from the table
shots = ['0', '4', '8', '16', '32', '64', '128']

# 'None' is used for missing values so matplotlib naturally skips them in the line
# xgboost = [None, 0.20, 0.20, 0.26, 0.26, 0.28, 0.34]
# lr = [None, 0.21, 0.22, 0.27, 0.24, 0.30, 0.31]
# ours = [None, 0.30, 0.26, 0.28, 0.26, 0.29, 0.38]

# # GPT4o is only evaluated at 0-shot
# gpt4o_score = 0.1732


xgboost = [None, 0.00, 0.29, 0.32, 0.33, 0.36, 0.40]
lr = [None, 0.28, 0.27, 0.33, 0.33, 0.35, 0.40]
ours = [None, 0.33, 0.31, 0.35, 0.32, 0.35, 0.45]

# GPT4o is only evaluated at 0-shot
gpt4o_score = 0.28

# Initialize the figure
plt.figure(figsize=(9, 6))

# Plot the few-shot methods with markers
plt.plot(shots, xgboost, marker='o', linestyle='-', linewidth=2, label='XGBoost')
plt.plot(shots, lr, marker='s', linestyle='-', linewidth=2, label='LR')
plt.plot(shots, ours, marker='^', linestyle='-', linewidth=2, label='Ours')

# Plot GPT4o as a single point at '0' and a dashed baseline across the chart
plt.scatter('0', gpt4o_score, color='red', zorder=5, label='GPT4o (zero-shot)')

# Format the axes and grid
plt.xlabel('Number of Shots', fontsize=12, fontweight='bold')
plt.ylabel('F1 score', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Adjust layout and display
plt.tight_layout()
plt.show()