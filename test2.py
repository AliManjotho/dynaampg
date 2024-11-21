import numpy as np
import matplotlib.pyplot as plt

# Generate 10 random samples
x = np.random.uniform(-5, 5, 10)
y = np.random.uniform(-10, 10, 10)

# Generate red samples with minimum distance constraint
red_x = []
red_y = []
attempts = 0
max_attempts = 1000  # Prevent infinite loop

while len(red_x) < 5 and attempts < max_attempts:
    new_x = np.random.uniform(-5, 5)
    new_y = np.random.uniform(-10, 10)
    
    # Calculate distances to all blue points
    distances = np.sqrt((x - new_x)**2 + (y - new_y)**2)
    
    # If minimum distance is >= 2, accept the point
    if np.min(distances) >= 2:
        red_x.append(new_x)
        red_y.append(new_y)
    
    attempts += 1

# Plot the scatter plots
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', marker='o', label='Blue Samples')
plt.scatter(red_x, red_y, color='red', marker='o', label='Red Samples')
plt.title('Scatter Plot of Randomly Generated Samples')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(-5, 5)
plt.ylim(-10, 10)
plt.legend()
plt.grid(True)
plt.show()
