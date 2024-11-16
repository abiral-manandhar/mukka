import numpy as np
import socket
import json
import time


def send_data(data):
    server_address = ('localhost', 6969)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    try:
        data_list = data.tolist() if isinstance(data, np.ndarray) else data
        json_data = json.dumps({"array": data_list})
        client_socket.sendall(json_data.encode('utf-8'))
    finally:
        client_socket.close()


def get_ip_address():
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
    try:
        parts = line.strip().split(':')
        if len(parts) != 2:
            raise ValueError("Line does not have exactly one ':' character.")
        key, values = parts
        values = values.strip()
        return np.array([float(x) for x in values.split(',')])
    except ValueError as e:
        print(f"Error parsing line '{line}': {e}")
        return None


def start_server():
    ip_address = get_ip_address()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip_address, 5000))
    server_socket.listen(1)

    # File to store print statements
    with open('Data/Stationary/output_log1.txt', 'w') as log_file:
        print(f"Server started on {ip_address}. Waiting for connections...")

        data_counter = 0
        data_interval = 1 / 17  # Interval for 17 data points per second
        start_time = None  # Initialize the start time as None
        first_data_received = False  # Flag to indicate when the first data is received
        duration_limit = 300  # Duration in seconds (5 minutes)

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

                    if not first_data_received:
                        # Start the timer when data starts arriving
                        start_time = time.perf_counter()
                        first_data_received = True

                    elapsed_time = time.perf_counter() - start_time if first_data_received else 0

                    if elapsed_time > duration_limit:  # Stop after the specified duration
                        print("Duration limit reached. Server is shutting down.")
                        log_file.write("Server shut down due to duration limit.\n")
                        client_socket.close()
                        server_socket.close()
                        return

                    data_str = data.decode('utf-8').strip()
                    if not data_str:
                        continue

                    lines = data_str.splitlines()

                    for line in lines:
                        line = line.strip()
                        if line.startswith("Accelerometer:"):
                            accel_data = parse_data_line(line)
                        elif line.startswith("Magnetometer:"):
                            mag_data = parse_data_line(line)
                        elif line.startswith("Gyroscope:"):
                            gyro_data = parse_data_line(line)

                    if gyro_data is not None and mag_data is not None and accel_data is not None:
                        log_file.write(
                            f"{elapsed_time:.8f} - Accel: {accel_data}, Mag: {mag_data}, Gyro: {gyro_data}\n")
                        print(f"{elapsed_time:.8f} - Accel: {accel_data}, Mag: {mag_data}, Gyro: {gyro_data}")
                        data_counter += 1
                    else:
                        print("Received data does not contain all required lines.")

                    # Throttle to 17 data points per second
                    time.sleep(data_interval)

                except Exception as e:
                    log_file.write(f"Unexpected error: {e}\n")
                    print(f"Unexpected error: {e}")

            client_socket.close()


if __name__ == "__main__":
    start_server()
