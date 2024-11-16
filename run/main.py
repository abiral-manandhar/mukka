from network import Network
from KalmanFilter import KalmanFilterQuaternion as kalman
import time
import socket
import numpy as np

def run():
    network = Network()
    ip_address = network.get_ip_address()
    data_interval = 1 / 17  # Interval for 17 data points per second
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip_address, 5000))
    server_socket.listen(1)

    print(f"Server started on {ip_address}. Waiting for connections...")

    # Initial quaternion and covariance
    q_init = np.array([0.17784454, -0.01484999, -0.00147595, 0.98394543])
    P_init = np.eye(4) * 0.01
    Q = np.eye(4) * 0.01
    R = np.eye(3) * 0.1

    # Create Kalman filter with gyro sensitivity set to 2.0 (or any desired value)
    kalman_filter = kalman(q_init, P_init, Q, R, gyro_sensitivity=58.0)

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established!")

        accel_data = None
        mag_data = None
        gyro_data = None

        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break

                data_str = data.decode('utf-8').strip()
                if not data_str:
                    continue

                lines = data_str.splitlines()

                for line in lines:
                    line = line.strip()
                    if line.startswith("Accelerometer:"):
                        accel_data = network.parse_data_line(line)
                    elif line.startswith("Magnetometer:"):
                        mag_data = network.parse_data_line(line)
                    elif line.startswith("Gyroscope:"):
                        gyro_data = network.parse_data_line(line)

                if accel_data is not None and gyro_data is not None:
                    dt = data_interval  # Time interval
                    kalman_filter.predict(gyro_data, dt)
                    kalman_filter.update(accel_data, mag_data)

                    quaternion = kalman_filter.q
                    w, x, y, z = quaternion
                    quaternion = [w, -x, -z, -y]  # Adjust for Unity's coordinate system
                    print(quaternion)
                    network.send_data(quaternion)

                else:
                    print("Received data does not contain all required lines.")

                time.sleep(data_interval)

            except Exception as e:
                print(f"Unexpected error: {e}")

        client_socket.close()

if __name__ == "__main__":
    run()
