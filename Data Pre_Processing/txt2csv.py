import pandas as pd

# Read the data from the file with timestamps
df_accel = pd.read_csv('accel_data_linear.txt', delim_whitespace=True, header=None)
df_gyro = pd.read_csv('gyro_data_linear.txt', delim_whitespace=True, header=None)
df_mag = pd.read_csv('mag_data_linear.txt', delim_whitespace=True, header=None)

# Assign column names, including timestamp
df_accel.columns = ['timestamp', 'accelX', 'accelY', 'accelZ']
df_gyro.columns = ['timestamp', 'gyroX', 'gyroY', 'gyroZ']
df_mag.columns = ['timestamp', 'magX', 'magY', 'magZ']

# Format the data to ensure consistent precision (e.g., 8 decimal places)
df_accel['accelX'] = df_accel['accelX'].map(lambda x: f"{x:.8f}")
df_accel['accelY'] = df_accel['accelY'].map(lambda x: f"{x:.8f}")
df_accel['accelZ'] = df_accel['accelZ'].map(lambda x: f"{x:.8f}")

df_gyro['gyroX'] = df_gyro['gyroX'].map(lambda x: f"{x:.8f}")
df_gyro['gyroY'] = df_gyro['gyroY'].map(lambda x: f"{x:.8f}")
df_gyro['gyroZ'] = df_gyro['gyroZ'].map(lambda x: f"{x:.8f}")

df_mag['magX'] = df_mag['magX'].map(lambda x: f"{x:.8f}")
df_mag['magY'] = df_mag['magY'].map(lambda x: f"{x:.8f}")
df_mag['magZ'] = df_mag['magZ'].map(lambda x: f"{x:.8f}")

# Save the DataFrame to a CSV file
df_accel.to_csv('accel_data_active_rot1.csv', index=False)
df_gyro.to_csv('gyro_data_active_rot1.csv', index=False)
df_mag.to_csv('mag_data_active_rot1.csv', index=False)

print(f"Data successfully converted and saved to accel_data_active_rot1.csv, gyro_data_active_rot1.csv, and mag_data_active_rot1.csv.")
