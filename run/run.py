import socket
import numpy as np
import json
import time
from network import Network
from KalmanFilter import KalmanFilterQuaternion as kalman


def send_data(quaternion, server_address=('localhost', 6969)):
    """Send quaternion data to a client."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(server_address)
            # Format quaternion as JSON: {"quaternion": [w, x, y, z]}
            json_data = json.dumps({"quaternion": quaternion})
            client_socket.sendall(json_data.encode('utf-8'))
    except Exception as e:
        print(f"Error sending data: {e}")



def get_ip_address():
    """Retrieve the local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.254.254.254', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    return ip_address


def parse_data_line(line):
    """Parse sensor data from a single line."""
    try:
        parts = line.strip().split(':')
        if len(parts) != 2:
            raise ValueError("Line does not have exactly one ':' character.")
        key, values = parts
        return np.array([float(x) for x in values.split(',')])
    except ValueError as e:
        print(f"Error parsing line '{line}': {e}")
        return None


def start_server():
    """Start the server to receive data and process it with the Kalman filter."""
    ip_address = get_ip_address()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip_address, 5000))
    server_socket.listen(1)

    print(f"Server started on {ip_address}. Waiting for connections...")

    # Kalman filter initialization
    q_init = np.array([0.17784454, -0.01484999, -0.00147595, 0.98394543])
    P_init = np.eye(4) * 0.01
    Q = np.eye(4) * 0.01
    R = np.eye(3) * 0.1
    kalman_filter = kalman(q_init, P_init, Q, R, gyro_sensitivity=58.0)

    data_interval = 1 / 17  # 17 data points per second
    duration_limit = 300  # Duration in seconds (5 minutes)

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established!")

        accel_data = None
        mag_data = None
        gyro_data = None
        start_time = time.perf_counter()

        while True:
            try:
                # Receive data
                data = client_socket.recv(1024)
                if not data:
                    break

                # Stop after duration limit
                elapsed_time = time.perf_counter() - start_time
                if elapsed_time > duration_limit:
                    print("Duration limit reached. Shutting down.")
                    client_socket.close()
                    server_socket.close()
                    return

                # Parse received data
                data_str = data.decode('utf-8').strip()
                if not data_str:
                    continue

                lines = data_str.splitlines()
                for line in lines:
                    if line.startswith("Accelerometer:"):
                        accel_data = parse_data_line(line)
                    elif line.startswith("Magnetometer:"):
                        mag_data = parse_data_line(line)
                    elif line.startswith("Gyroscope:"):
                        gyro_data = parse_data_line(line)

                # Process and send quaternion if all data is available
                if accel_data is not None and gyro_data is not None:
                    dt = data_interval
                    kalman_filter.predict(gyro_data, dt)

                    if mag_data is not None:
                        kalman_filter.update(accel_data, mag_data)

                    quaternion = kalman_filter.q
                    # Adjust quaternion to Unity's expected coordinate system
                    quaternion = [quaternion[0], -quaternion[1], -quaternion[3], -quaternion[2]]

                    print(f"Processed Quaternion: {quaternion}")
                    send_data(quaternion)  # Send the formatted quaternion

                    # Reset processed data
                    accel_data = mag_data = gyro_data = None

                else:
                    print("Incomplete data received. Waiting for next set.")

                time.sleep(data_interval)

            except Exception as e:
                print(f"Unexpected error: {e}")
                break

        client_socket.close()


if __name__ == "__main__":
    start_server()
