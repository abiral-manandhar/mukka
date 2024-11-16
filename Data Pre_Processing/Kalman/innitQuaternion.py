import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calculate_initial_orientation(accel, mag):
    """
    Calculate the initial quaternion orientation based on accelerometer and magnetometer data.
    :param accel: numpy array of accelerometer data [ax, ay, az]
    :param mag: numpy array of magnetometer data [mx, my, mz]
    :return: numpy array representing the initial quaternion [q0, q1, q2, q3]
    """

    # Normalize accelerometer and magnetometer data
    accel = normalize(accel)
    mag = normalize(mag)

    # Calculate pitch (rotation around x-axis) and roll (rotation around y-axis) from accelerometer
    ax, ay, az = accel
    pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2))  # Rotation around x-axis
    roll = np.arctan2(-ay, az)  # Rotation around y-axis

    # Calculate yaw (rotation around z-axis) from magnetometer
    mx, my, mz = mag

    # Adjust magnetometer readings based on pitch and roll
    mag_x = mx * np.cos(pitch) + mz * np.sin(pitch)
    mag_y = mx * np.sin(roll) * np.sin(pitch) + my * np.cos(roll) - mz * np.sin(roll) * np.cos(pitch)

    # Calculate yaw (heading) from adjusted magnetometer values
    yaw = np.arctan2(-mag_y, mag_x)

    # Convert Euler angles (pitch, roll, yaw) to quaternion
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])

# Example data (replace with actual sensor data)
accel_data = np.array([0.28395,0.081,9.889951])  # Assuming the device is stationary and aligned with gravity
mag_data = np.array([-23.34375,-8.8125,-32.8875 ])  # Example magnetometer data

# Calculate the initial orientation quaternion
initial_quaternion = calculate_initial_orientation(accel_data, mag_data)
print("Initial Quaternion:", initial_quaternion)

# Now you can use initial_quaternion to feed into the Kalman filter as the initial orientation
