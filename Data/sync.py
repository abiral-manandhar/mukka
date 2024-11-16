import pandas as pd

# Load the data
data = pd.read_csv('../Data/Stationary/sensor_data.csv')

# Extract individual sensor data
timestamps = data['timestamp']
gyro_data = data[['gyro_x', 'gyro_y', 'gyro_z']]
accel_data = data[['acc_x', 'acc_y', 'acc_z']]
mag_data = data[['mag_x', 'mag_y', 'mag_z']]

# Assuming you have a stationary data file loaded similarly
stationary_data = pd.read_csv('../Data/Stationary/sensor_data.csv')

# Calculate the bias (mean values during stationary period)
gyro_bias = stationary_data[['gyro_x', 'gyro_y', 'gyro_z']].mean()
accel_bias = stationary_data[['acc_x', 'acc_y', 'acc_z']].mean()
mag_bias = stationary_data[['mag_x', 'mag_y', 'mag_z']].mean()

# # Apply the calibration to the active data aka data that is coming in rn
# gyro_calibrated = gyro_data - gyro_bias
# accel_calibrated = accel_data - accel_bias
# mag_calibrated = mag_data - mag_bias
