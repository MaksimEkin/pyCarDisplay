"""

"""
class KalmanFilter(object):
    """

    """
    def __init__(self, P = 1, H = 1, F = 1, Q = 0.1):
        self.P = P
        self.H = H
        self.F = F
        self.Q = Q

    def Predict(self, acceleration, speed_previous, delta_time):
        """

        Parameters
        ----------
        acceleration :
        speed_previous :
        delta_time :

        Returns
        -------

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
        speed_data_from_sensor :
        R :
        speed_predict :

        Returns
        -------

        """
        y = speed_data_from_sensor - self.H * speed_predict
        S = self.H * self.P * self.H + R
        K = self.P * self.H * S ** (-1)

        speed_fused = speed_predict + K * y
        self.P = (1 - K * self.H) * self.P

        return speed_fused
