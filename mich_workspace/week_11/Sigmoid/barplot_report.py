import matplotlib.pyplot as plt
import numpy as np

# Define the models and their performance for each input situation
models = ['Full GRU', 'Simple GRU', 'CB-GRU', 'CB-RNN-tied', 'Dale-CB', 'CB-RNN-tied-STP', 'Dale-CB-STP', 'Vanilla RNN']
input_situations = ['Sigmoid']
performance = np.array([
    [92.15], # Full GRU
    [90.96], # Simple GRU
    [80.10], # CB-GRU
    [80.91], # CB-RNN-tied
    [78.76], # Dale-CB
    [77.01], # CB-RNN-tied-STP
    [56.53],  # Dale-CB-STP
    [10.28] # Vanilla RNN
])

# Set the width of each bar
bar_width = 0.35

# Set the positions of the bars on the x-axis
x = np.arange(len(models))
# Add a baseline level of 10 and sketch it on the graph
baseline = np.array([10 for _ in range(len(models))])

# Create the bar plot
# Adjusting the plot to make it more visually appealing
fig, ax = plt.subplots(figsize=(14, 8))
plt.rcParams.update({'font.size': 13})

# Normalize data for better comparison and baseline at 10%
normalized_performance = performance / 100.0
baseline = 0.10  # 10% baseline

# Create the bar plot, centering the bars
rects1 = ax.bar(x, normalized_performance[:, 0], bar_width, label='Performance ($\phi$=Sigmoid)', color='skyblue')

# Draw the baseline
ax.axhline(y=baseline, color='black', linestyle='--', label='Baseline (10%)')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Normalized Performance')
ax.set_title('Comparison of Models Performances')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right", fontsize=12)
ax.legend(loc='upper right')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.95, 'Input size = 8\nStride = 4\nHidden size = 24', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# Adding value labels on top of each bar
for rect in rects1:
    height = rect.get_height()
    # set font size of annotation
    
    ax.annotate(f'{height:.2%}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adjust the y-axis to show percentages
ax.set_yticklabels([f'{y:.0%}' for y in ax.get_yticks()])

# Make the layout more compact
plt.tight_layout()

# Show the plot
plt.show()
