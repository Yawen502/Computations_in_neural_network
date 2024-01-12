import matplotlib.pyplot as plt
import numpy as np

'''
- full GRU
- simple GRU
- CB-GRU
- STP nonDaleCB
- DaleCB
- STPDaleCB
- vanilla RNN
'''
# Define the models and their performance for each input situation
models = ['CB-GRU', 'CB-RNN', 'CB-RNN-tied', 'Dale-CB', 'CB-GRU-STP', 'CB-RNN-STP', 'CB-RNN-tied-STP', 'Dale-CB-STP']
input_situations = ['Sigmoid', 'ReLU']
performance = np.random.rand(len(models), len(input_situations))

# 01 CB-GRU
performance[0,:] = [80.10, 83.09]
# 02 CB-RNN
performance[1,:] = [85.44, 86.88]
# 03 CB-RNN-tied
performance[2,:] = [80.91, 79.84]
# 04 Dale-CB
performance[3,:] = [81.93, 77.26]
# 05 CB-GRU-STP
performance[4,:] = [80.93, 9.80 ]
# 06 CB-RNN-STP
performance[5,:] = [81.37, 9.80 ]
# 07 CB-RNN-tied-STP
performance[6,:] = [77.01, 9.80 ]
# 08 Dale-CB-STP
performance[7,:] = [75.08, 9.80 ]
# Set the width of each bar
bar_width = 0.2

# Set the positions of the bars on the x-axis
x = np.arange(len(models))

# Create the bar plot
fig, ax = plt.subplots()
for i, situation in enumerate(input_situations):
    ax.bar(x + i * bar_width, performance[:, i], bar_width, label=f'{situation}')


# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Performance')
ax.set_title('Comparison of Models')
ax.set_xticks(x + bar_width * (len(input_situations) - 1) / 2)
ax.set_xticklabels(models)
ax.legend()

# Show the plot
plt.show()

# stride length 4
# input length 4, Accuracy of the model: 9.58%
# input length 8, Accuracy of the model: 18.13%
# input length 12, Accuracy of the model: 9.8%