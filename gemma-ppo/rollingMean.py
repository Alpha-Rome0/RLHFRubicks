import numpy as np
import matplotlib.pyplot as plt

file = open("output.txt", "r")
lines = file.readlines()

rewards = []
rollingMean = []
for line in lines:
    idx = line.find("reward: ")
    if idx != -1:
        rewards.append(int(line[idx+8:].strip()))
        rollingMean.append(np.mean(np.array(rewards)))

# Generate x values (0 to len(numbers)-1)
x_values = range(len(rollingMean))

# Create the plot
plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.plot(x_values, rollingMean, marker='o')  # Plot x and y using a line and markers

# Adding title and labels
plt.title('Average Reward After Every Training Step')
plt.xlabel('Training Step')
plt.ylabel('Average Reward')

# Optionally, you can add grid lines for better readability
plt.grid(True)

# Show the plot
plt.savefig("movingAverageDistanceRewardFunction")