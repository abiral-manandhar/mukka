import numpy as np
import socket
import json


class MadgwickAHRS:
    def __init__(self, sample_period, beta=0.1):
        self.sample_period = sample_period
        self.beta = beta
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion

    def update(self, gyro, accel, mag):
        q = self.quaternion
        sample_period = self.sample_period
        beta = self.beta

        # Normalize accelerometer measurement
        if np.linalg.norm(accel) == 0:
            return  # handle NaN
        accel = accel / np.linalg.norm(accel)

        # Normalize magnetometer measurement
        if np.linalg.norm(mag) == 0:
            return  # handle NaN
        mag = mag / np.linalg.norm(mag)

        # Reference direction of Earth's magnetic field
        h = self._quaternion_multiply(q, self._quaternion_multiply(np.concatenate([[0], mag]),
                                                                   self._quaternion_conjugate(q)))
        b = np.array([0, np.linalg.norm(h[1:3]), 0, h[3]])

        # Gradient descent algorithm corrective step
        F = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]) - accel[0],
            2 * (q[0] * q[1] + q[2] * q[3]) - accel[1],
            2 * (0.5 - q[1] ** 2 - q[2] ** 2) - accel[2],
            2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - mag[0],
            2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - mag[1],
            2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2) - mag[2]
        ])
        J = np.array([
            [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
            [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
            [0, -4 * q[1], -4 * q[2], 0],
            [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0], -4 * b[1] * q[3] + 2 * b[3] * q[1]],
            [-2 * b[1] * q[3] + 2 * b[3] * q[1], 2 * b[1] * q[2] + 2 * b[3] * q[0], 2 * b[1] * q[1] + 2 * b[3] * q[3],
             -2 * b[1] * q[0] + 2 * b[3] * q[2]],
            [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2], 2 * b[1] * q[1]]
        ])
        step = J.T @ F
        step = step / np.linalg.norm(step)  # Normalize step magnitude

        # Compute rate of change of quaternion
        q_dot = 0.5 * self._quaternion_multiply(q, np.concatenate([[0], gyro])) - beta * step

        # Integrate to yield quaternion
        q = q + q_dot * sample_period
        self.quaternion = q / np.linalg.norm(q)  # Normalize quaternion

    def _quaternion_multiply(self, q, r):
        return np.array([
            q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
            q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
            q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
            q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0]
        ])

    def _quaternion_conjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])


def send_data(data, client_socket):
    try:
        # Serialize list to JSON
        json_data = json.dumps({"quaternion": data.tolist()})
        client_socket.sendall(json_data.encode('utf-8'))
    except Exception as e:
        print(f"Error sending data: {e}")


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # Doesn't even need to be reachable
        s.connect(('10.254.254.254', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    return ip_address


def start_server():
    ip_address = get_ip_address()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip_address, 5000))
    server_socket.listen(1)
    print(f"Server started on {ip_address}. Waiting for connections...")

    sample_period = 1 / 256  # Sample period of 256 Hz
    madgwick = MadgwickAHRS(sample_period)

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established!")

        accel_data = []
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
                        accel = np.array([float(x) for x in line.split(':')[1].split(',')])
                        accel_data.append(accel)
                    elif line.startswith("Magnetometer:"):
                        mag_data = np.array([float(x) for x in line.split(':')[1].split(',')])
                    elif line.startswith("Gyroscope:"):
                        gyro_data = np.array([float(x) for x in line.split(':')[1].split(',')])

                if gyro_data is not None and mag_data is not None and accel_data:
                    # Use the most recent accelerometer data
                    accel = accel_data[-1]
                    print(f"Received data - Accel: {accel}, Mag: {mag_data}, Gyro: {gyro_data}")
                    madgwick.update(gyro_data, accel, mag_data)
                    quaternion = madgwick.quaternion
                    send_data(quaternion, client_socket)
                    print(f"Processed and sent quaternion: {quaternion}")

            except ValueError as e:
                print(f"Error parsing data: {e}")
                print(f"Received data: {data_str}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        client_socket.close()


if __name__ == "__main__":
    start_server()
