import pandas as pd

# Read the data from the file
df = pd.read_csv('accel_data_linear.txt', delim_whitespace=True, header=None)
df1 = pd.read_csv('gyro_data_linear.txt', delim_whitespace=True, header=None)
df2 = pd.read_csv('mag_data_linear.txt', delim_whitespace=True, header=None)

# Function to format numbers to a fixed number of decimal places
def format_number(num, decimals=8):
    return f"{num:.{decimals}f}"

# Apply formatting to each element in the DataFrame
for col in df.columns:
    df[col] = df[col].apply(lambda x: format_number(x, decimals=8))

# Write the formatted DataFrame back to a file
df.to_csv('accel_data_active_rot3.txt', index=False, header=False, sep=' ')
df1.to_csv('gyro_data_active_rot3.txt', index=False, header=False, sep=' ')
df2.to_csv('mag_data_active_rot3.txt', index=False, header=False, sep=' ')
