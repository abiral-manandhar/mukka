import pandas as pd

# Read the data from the files
df_accel = pd.read_csv('accel_data_linear.txt', delim_whitespace=True, header=None)
df_gyro = pd.read_csv('gyro_data_linear.txt', delim_whitespace=True, header=None)
df_mag = pd.read_csv('mag_data_linear.txt', delim_whitespace=True, header=None)

# Function to format numbers to a fixed number of decimal places
def format_number(num, decimals=8):
    return f"{num:.{decimals}f}"

# Apply formatting to each element in each DataFrame
for col in df_accel.columns:
    df_accel[col] = df_accel[col].apply(lambda x: format_number(x, decimals=8))

for col in df_gyro.columns:
    df_gyro[col] = df_gyro[col].apply(lambda x: format_number(x, decimals=8))

for col in df_mag.columns:
    df_mag[col] = df_mag[col].apply(lambda x: format_number(x, decimals=8))

# Write the formatted DataFrames back to files
df_accel.to_csv('accel_data_active_rot1.txt', index=False, header=False, sep=' ')
df_gyro.to_csv('gyro_data_active_rot1.txt', index=False, header=False, sep=' ')
df_mag.to_csv('mag_data_active_rot1.txt', index=False, header=False, sep=' ')

print("Data successfully formatted and saved.")
