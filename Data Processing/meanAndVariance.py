import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../Data/Stationary/accel.csv')


# Calculate rolling mean and variance to observe stabilization
window_size = 10  # Adjust this depending on your sampling rate and the noise level
data['accelX_mean'] = data['accelX'].rolling(window=window_size).mean()
data['accelX_var'] = data['accelX'].rolling(window=window_size).var()
data['accelY_mean'] = data['accelY'].rolling(window=window_size).mean()
data['accelY_var'] = data['accelY'].rolling(window=window_size).var()
data['accelZ_mean'] = data['accelZ'].rolling(window=window_size).mean()
data['accelZ_var'] = data['accelZ'].rolling(window=window_size).var()

# Plot the data
plt.figure(figsize=(14, 10))

# Plot accelX Data
plt.subplot(3, 1, 1)
plt.plot(data.index, data['accelX'], label='accelX Raw Data', color='blue')
plt.plot(data.index, data['accelX_mean'], label='accelX Mean', color='orange')
plt.plot(data.index, data['accelX_var'], label='accelX Variance', color='green')
plt.axvline(x=40, color='red', linestyle='--', label='40 sec mark')
plt.title('accelX: Raw Data, Mean, and Variance')
plt.xlabel('Sample Index')
plt.ylabel('accelX Value')
plt.legend()

# Plot accelY Data
plt.subplot(3, 1, 2)
plt.plot(data.index, data['accelY'], label='accelY Raw Data', color='blue')
plt.plot(data.index, data['accelY_mean'], label='accelY Mean', color='orange')
plt.plot(data.index, data['accelY_var'], label='accelY Variance', color='green')
plt.axvline(x=40, color='red', linestyle='--', label='40 sec mark')
plt.title('accelY: Raw Data, Mean, and Variance')
plt.xlabel('Sample Index')
plt.ylabel('accelY Value')
plt.legend()

# Plot accelZ Data
plt.subplot(3, 1, 3)
plt.plot(data.index, data['accelZ'], label='accelZ Raw Data', color='blue')
plt.plot(data.index, data['accelZ_mean'], label='accelZ Mean', color='orange')
plt.plot(data.index, data['accelZ_var'], label='accelZ Variance', color='green')
plt.axvline(x=40, color='red', linestyle='--', label='40 sec mark')
plt.title('accelZ: Raw Data, Mean, and Variance')
plt.xlabel('Sample Index')
plt.ylabel('accelZ Value')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate rolling variance with a larger window for a specific segment
data['accelY_var_large'] = data['accelY'].rolling(window=4889).var()

# Save the variance data to a CSV file
variance_data = data[['accelX_var', 'accelY_var', 'accelZ_var']]
variance_data.to_csv('variance_data.csv', index=False)

# Compute the mean of each variance column
mean_accelX_var = data['accelX_var'].mean()
mean_accelY_var = data['accelY_var'].mean()
mean_accelZ_var = data['accelZ_var'].mean()

# Print the mean variance values
print(f"Mean accelX Variance: {mean_accelX_var}")
print(f"Mean accelY Variance: {mean_accelY_var}")
print(f"Mean accelZ Variance: {mean_accelZ_var}")

# Save the mean variance values into an array
mean_variances = np.array([mean_accelX_var, mean_accelY_var, mean_accelZ_var])
print("Mean Variances Array:", mean_variances)

var = pd.read_csv('../Data/Stationary/variance_data.csv')

new_Data = data['accelX'] - var["accelX_var"]
