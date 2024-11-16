import numpy as np
from ahrs.filters import Madgwick

# Initialize Madgwick filter with a sampling rate of 17 Hz and beta
madgwick_filter = Madgwick(frequency=17, beta=0.3)


def get_orientation(accel_data, gyro_data, mag_data):

    accel = np.array(accel_data)
    gyro = np.array(gyro_data)
    mag = np.array(mag_data)

    # Fuse the sensor data and get the quaternion
    quaternion = madgwick_filter.updateIMU(gyro=gyro, acc=accel, mag=mag)

    return quaternion


# Example usage with real-time data input
while True:
    # Simulate reading real-time sensor data (replace this with actual sensor readings)
    accel_data = [0.0, 0.1, -1.0]  # Replace with live accelerometer data
    gyro_data = [0.01, 0.02, 0.01]  # Replace with live gyroscope data
    mag_data = [0.3, 0.2, 0.4]  # Replace with live magnetometer data

    # Get the quaternion
    orientation = get_orientation(accel_data, gyro_data, mag_data)

    # Print the quaternion
    print("Quaternion: ", orientation)
