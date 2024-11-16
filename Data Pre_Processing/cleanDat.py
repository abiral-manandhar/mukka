import re

# Input file
input_file = '../Data/Stationary/output_log1.txt'

# Output files
accel_file = 'accel_data_linear.txt'
gyro_file = 'gyro_data_linear.txt'
mag_file = 'mag_data_linear.txt'

# Regular expressions to match each type of data with timestamps
timestamp_pattern = re.compile(r'([0-9]+\.[0-9]+)')  # To capture timestamps
accel_pattern = re.compile(r'Accel:\s*\[\s*([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)\s*\]')
gyro_pattern = re.compile(r'Gyro:\s*\[\s*([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)\s*\]')
mag_pattern = re.compile(r'Mag:\s*\[\s*([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)\s*\]')

# Open output files
with open(accel_file, 'w') as accel_out, open(gyro_file, 'w') as gyro_out, open(mag_file, 'w') as mag_out:
    # Read input file line by line
    with open(input_file, 'r') as infile:
        for line in infile:
            # Extract timestamp
            timestamp_match = timestamp_pattern.search(line)
            timestamp = timestamp_match.group(1) if timestamp_match else None

            # Find and write accelerometer data
            accel_match = accel_pattern.search(line)
            if accel_match:
                accel_out.write(f"{timestamp} {accel_match.group(1)} {accel_match.group(2)} {accel_match.group(3)}\n")

            # Find and write gyroscope data
            gyro_match = gyro_pattern.search(line)
            if gyro_match:
                gyro_out.write(f"{timestamp} {gyro_match.group(1)} {gyro_match.group(2)} {gyro_match.group(3)}\n")

            # Find and write magnetometer data
            mag_match = mag_pattern.search(line)
            if mag_match:
                mag_out.write(f"{timestamp} {mag_match.group(1)} {mag_match.group(2)} {mag_match.group(3)}\n")

print("Data has been separated into accel_data_linear.txt, gyro_data_linear.txt, and mag_data_linear.txt.")
