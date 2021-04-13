"""
This file simulates an IMU sensor. Data must be loaded from kitti_data_loader_api.py
and passed to this module.
"""
import numpy as np
import pandas as pd
import warnings
import sys


class IMU:
    def __init__(self, data, verbose=True, R_covariance=0.1, random_state=42):
        """
        Initilie the IMU sensor.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas DataFrame of IMU data.
        verbose : bool, optional
            Verbosity flag. The default is True.
        R_covariance : float, optional
            Covariance for the random error. The default is 0.1.
        random_state : bool, optional
            random seed. The default is 42.

        Returns
        -------
        None.

        """

        assert type(
            data) is pd.DataFrame, "data should be type Pandas DataFrame"
        self.imu_data = data
        self.num_frames = len(self.imu_data.frame.unique())
        self.params_avail = list(self.imu_data.columns.values)
        self.frame = 0
        self.verbose = verbose
        self.R_covariance = R_covariance
        np.random.seed(random_state)
        if self.verbose:
            print("Initilized IMU")
            print("Current frame:" + str(self.frame))
            print("Total number of frames:" + str(len(self.imu_data)))

    def get_R(self):
        """
        Returns the error covariance.

        Returns
        -------
        float
            Error covariance.

        """
        return self.R_covariance

    def get_avail_data(self):
        """
        Returns the available IMU features, i.e. column names.

        Returns
        -------
        list
            names of the IMU columns.

        """
        return list(self.imu_data.columns)

    def get_params(self):
        """
        Returns the object variables.

        Returns
        -------
        dict
            self variables.

        """
        return vars(self)

    def read_sensor(self, name=None, add_noise=True, advance_frame=True):
        """
        Simulates reading one IMU sensor data and increments the frame.

        Parameters
        ----------
        name : str, optional
            Specify to read specific IMU data. The default is None.
        add_noise : bool, optional
            If True, error is added to the data. The default is True.
        advance_frame : bool, optional
            If True, frame is advanced to the next one. The default is True.

        Returns
        -------
        dict
            {"data": data, "noise": noise, "true": data-noise}.

        """

        if self.frame > self.num_frames:
            if self.verbose:
                warnings.warn(
                    "Reached at the end of available number of frames. Reverting to frame 0. Number of frames: " + str(self.num_frames))
            self.frame = 0

        noise = 0

        data = None
        if name is None:

            if add_noise:
                num_columns = len(self.params_avail) - \
                    1  # -1 for the frame column
                noise = np.random.normal(0, self.R_covariance, num_columns)
                data = self.imu_data[self.imu_data.frame ==
                                     self.frame].loc[:, self.imu_data.columns != "frame"] + noise
            else:
                data = self.imu_data[self.imu_data.frame ==
                                     self.frame].loc[:, self.imu_data.columns != "frame"]
        else:
            self.__check_param(name)
            if add_noise:
                noise = np.random.normal(0, self.R_covariance, 1)
            data = self.imu_data[self.imu_data.frame ==
                                 self.frame][str(name)].values + noise

        if advance_frame:
            self.frame += 1
        return {"data": data, "noise": noise, "true": data-noise}

    def get_column(self, name: str):
        """
        Returns all the frame data for one column.

        Parameters
        ----------
        name : str
            column name.

        Returns
        -------
        pd.DataFrame
            one column from the dataframe.

        """
        self.__check_param(name)
        return np.array(self.imu_data[str(name)])

    def set_frame(self, frame: int):
        """
        manually set the frame,

        Parameters
        ----------
        frame : int
            frame number.

        Returns
        -------
        None.

        """
        if frame > self.num_frames or frame < 0:
            sys.exit("Frame has to be a positive intiger, and has to be less than the total number of available frames. Number of frames: " + str(self.num_frames))
        self.frame = frame

    def __check_param(self, name):
        """
        Ensures the requested name is available in the DataFrame

        Parameters
        ----------
        name : str
            requested name.

        Returns
        -------
        None.

        """
        assert name in self.params_avail, "Parameter is not available in IMU. Parameters: " + \
            ", ".join(self.params_avail)
