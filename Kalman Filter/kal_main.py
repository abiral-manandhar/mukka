import numpy as np

#State Vector-The shawty we estimating
drift = np.zeros(6)  # For example, [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]

# State transition matrix (F)
F = np.eye(6)

# Measurement matrix (H) - assumes you measure angles and angular rates directly
H = np.eye(6)

# Covariance matrices
P = np.eye(6) * 1000  # Initial uncertainty in the state estimate
Q = np.eye(6) * 0.01  # Process noise covariance
R = np.eye(6) * 0.1   # Measurement noise covariance

def predict(x, P, F, Q):
    # Predict the next state
    x = F @ x
    # Predict the next covariance
    P = F @ P @ F.T + Q
    return x, P

def update(x, P, z, H, R):
    # Measurement residual
    y = z - H @ x
    # Residual covariance
    S = H @ P @ H.T + R
    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)
    # Update the state estimate
    x = x + K @ y
    # Update the covariance estimate
    P = (np.eye(len(P)) - K @ H) @ P
    return x, P
