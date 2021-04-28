"""
Kalman filter definition to smooth out noise in sensor readings
"""
import numpy as np


class KalmanFilter(object):
    def __init__(self, P=1, H=1, F=1, Q=0.1):
        """

        Parameters
        ----------
        P : covariance of the observation noise
        H : observation model
        F : state-transition model
        Q : covariance of the process noise

        Returns
        -------

        """
        self.P = P
        self.H = H
        self.F = F
        self.Q = Q

    def Predict(self, acceleration, speed_previous, delta_time):
        """

        Parameters
        ----------
        acceleration : new data reading
        speed_previous : previous data point
        delta_time : change in time from last sensor reading

        Returns
        -------
        speed_predict : predicted data output
        """
        # Prediction Model: speed = F * previous speed + acceleration * delata_time
        speed_predict = self.F * speed_previous + acceleration * delta_time

        # Equation: P = F * P * F' + Q
        self.P = self.F * self.P * self.F + np.random.normal(0, self.Q, 1)

        return speed_predict

    def Update(self, speed_data_from_sensor, R, speed_predict):
        """

        Parameters
        ----------
        speed_data_from_sensor : sensor data read
        R : covariance
        speed_predict : predicted datapoint

        Returns
        -------
        speed_fused : data point thathas been filtered
        """
        y = speed_data_from_sensor - self.H * speed_predict
        S = self.H * self.P * self.H + R
        K = self.P * self.H * S ** (-1)

        speed_fused = speed_predict + K * y
        self.P = (1 - K * self.H) * self.P

        return speed_fused
