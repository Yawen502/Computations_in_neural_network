import matplotlib.pyplot as plt
import numpy as np

# Define the models and their performance for each input situation
models = ['CB-GRU', 'CB-RNN', 'CB-RNN-tied', 'Dale-CB', 'CB-GRU-STP', 'CB-RNN-STP', 'CB-RNN-tied-STP', 'Dale-CB-STP']
input_situations = ['Sigmoid', 'ReLU']
performance = np.array([
    [80.10, 83.09], # CB-GRU
    [85.44, 86.88], # CB-RNN
    [80.91, 79.84], # CB-RNN-tied
    [78.76, 77.26], # Dale-CB
    [80.93, 9.80 ], # CB-GRU-STP
    [81.37, 9.80 ], # CB-RNN-STP
    [77.01, 9.80 ], # CB-RNN-tied-STP
    [56.53, 9.80 ]  # Dale-CB-STP
])

# Set the width of each bar
bar_width = 0.35

# Set the positions of the bars on the x-axis
x = np.arange(len(models))

# Create the bar plot
fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - bar_width/2, performance[:, 0], bar_width, label='Sigmoid')
rects2 = ax.bar(x + bar_width/2, performance[:, 1], bar_width, label='ReLU')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Performance')
ax.set_title('Comparison of Models by Activation Function')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)
ax.legend()

# Function to add labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call the function to add labels
autolabel(rects1)
autolabel(rects2)

# Make the layout more compact
plt.tight_layout()

# Show the plot
plt.show()
