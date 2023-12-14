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
models = ['Full GRU', 'Simple GRU', 'CB-GRU', 'STPCB', 'DaleCB', 'STPDaleCB', 'Vanilla RNN']
input_situations = [4, 8, 16]
performance = np.random.rand(len(models), len(input_situations))
# 01 Full GRU
performance[0,:] = [88.74, 87.28, 92.32]
# 02 Simple GRU
performance[1,:] = [89.72, 89.59, 91.71]
# 03 CB-GRU
performance[2,:] = [85.0, 79.69, 80.45]
# 04 STPnonDaleCB
performance[3,:] = [ , 73.78, 51.69]

# 05 STPDaleCB
performance[4,:] = [46.5,,]
# 06 DaleCB
performance[4,:] = [ , , ]
# 07 Vanilla RNN
performance[6,:] = [9.58, 18.13, 9.8]
# Set the width of each bar
bar_width = 0.2

# Set the positions of the bars on the x-axis
x = np.arange(len(models))

# Create the bar plot
fig, ax = plt.subplots()
for i, situation in enumerate(input_situations):
    ax.bar(x + i * bar_width, performance[:, i], bar_width, label=f'Input = {situation}')


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