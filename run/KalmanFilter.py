import numpy as np

class KalmanFilterQuaternion:
    def __init__(self, q_init, P_init, Q, R, gyro_sensitivity=1.0):
        self.q = q_init  # Initial quaternion
        self.P = P_init  # Initial error covariance
        self.Q = Q       # Process noise covariance
        self.R = R       # Measurement noise covariance
        self.gyro_sensitivity = gyro_sensitivity  # Sensitivity multiplier for the gyroscope

    def predict(self, gyro, dt):
        # Apply the sensitivity factor to the gyroscope data
        gyro = self.gyro_sensitivity * gyro
        F = self._state_transition_matrix(gyro, dt)
        self.q = self._normalize(F @ self.q)  # Predict next state
        self.P = F @ self.P @ F.T + self.Q  # Update covariance matrix

    def update(self, accel, mag=None):
        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel)  # Normalize accel data

        H = self._measurement_matrix(self.q)  # Measurement matrix
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)  # Kalman gain

        z = self._measurement(accel)  # Measurement (accel)
        y = z - self._predict_measurement(self.q)  # Innovation

        self.q = self._normalize(self.q + K @ y)  # Update state
        self.P = (np.eye(4) - K @ H) @ self.P  # Update covariance

    def _state_transition_matrix(self, gyro, dt):
        gx, gy, gz = np.radians(gyro)
        F = np.array([
            [1, -0.5*gx*dt, -0.5*gy*dt, -0.5*gz*dt],
            [0.5*gx*dt, 1, 0.5*gz*dt, -0.5*gy*dt],
            [0.5*gy*dt, -0.5*gz*dt, 1, 0.5*gx*dt],
            [0.5*gz*dt, 0.5*gy*dt, -0.5*gx*dt, 1]
        ])
        return F

    def _measurement_matrix(self, q):
        q1, q2, q3, q4 = q
        return np.array([
            [-2*q3, 2*q4, -2*q1, 2*q2],
            [2*q2, 2*q1, 2*q4, 2*q3],
            [0, -4*q2, -4*q3, 0]
        ])

    def _measurement(self, accel):
        ax, ay, az = accel
        return np.array([ax, ay, az])

    def _predict_measurement(self, q):
        q1, q2, q3, q4 = q
        return np.array([
            2 * (q2 * q4 - q1 * q3),
            2 * (q1 * q2 + q3 * q4),
            2 * (0.5 - q2 ** 2 - q3 ** 2)
        ])

    def _normalize(self, q):
        return q / np.linalg.norm(q)
