import socket
import json
import numpy as np

class Network:
    def __init__(self, server_ip='localhost', server_port=6969):
        self.server_address = (server_ip, server_port)

    def send_data(self, data):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(self.server_address)

        try:
            data_list = data.tolist() if isinstance(data, np.ndarray) else data
            json_data = json.dumps({"quaternion": data_list})
            client_socket.sendall(json_data.encode('utf-8'))
        finally:
            client_socket.close()

    def get_ip_address(self):
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

    def parse_data_line(self, line):
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

