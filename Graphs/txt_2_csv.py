import csv
import numpy as np

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        # Split the line into components and convert to float
        values = line.strip().split()
        if len(values) == 3:
            data.append([float(val) for val in values])
    return data

def generate_timestamps(duration, sampling_rate):
    num_samples = int(duration * sampling_rate)
    timestamps = np.linspace(0, duration, num_samples, endpoint=False)
    return timestamps

def write_to_csv(output_file, gyro_data, accel_data, mag_data, timestamps):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z', 'mag_x', 'mag_y', 'mag_z'])
        for timestamp, gyro, accel, mag in zip(timestamps, gyro_data, accel_data, mag_data):
            writer.writerow([timestamp] + gyro + accel + mag)

# File paths
gyro_file = '../Data/Active Data(Rotational)/rot2/gyro_data_active_rot2.txt'
accel_file = '../Data/Active Data(Rotational)/rot2/accel_data_active_rot2.txt'
mag_file = '../Data/Active Data(Rotational)/rot2/mag_data_active_rot2.txt'
output_csv = '../Data/Active Data(Rotational)/rot2/sensors_rot2.csv'

# Read data from text files
gyro_data = read_text_file(gyro_file)
accel_data = read_text_file(accel_file)
mag_data = read_text_file(mag_file)

# Generate timestamps
duration = 100  # 5 minutes = 300 seconds
sampling_rate = len(gyro_data) / duration  # Assuming all files have the same number of samples
timestamps = generate_timestamps(duration, sampling_rate)

# Write to CSV
write_to_csv(output_csv, gyro_data, accel_data, mag_data, timestamps)
