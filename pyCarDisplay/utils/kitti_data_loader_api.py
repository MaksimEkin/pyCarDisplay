"""
Parses the Kitti dataset's IMU data into a Pandas DataFrame format.
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime


class DataLoader(object):
    def __init__(self, path_lidar="", path_imu=""):
        """
        Initilize the parser.

        Parameters
        ----------
        path_lidar : str, optional
            path to the lidar data. The default is "".
        path_imu : str, optional
            path to the imu data. The default is "".

        Returns
        -------
        None.

        """
        # IMU columns
        self.imu_columns = [
            "lat", "lon", "alt", "roll", "pitch", "yaw", "vn", "ve", "vf",
            "vl", "vu", "ax", "ay", "az", "af", "al", "au", "wx", "wy", "wz",
            "wf", "wl", "wu", "pos_accuracy", "vel_accuracy", "navstat",
            "numsats", "posmode", "velmode", "orimode"
        ]

        self.path_lidar = path_lidar
        self.path_imu = path_imu

        self.lidar_data = None
        self.imu_data = None

    def load_lidar(self, sample_size=1):
        """
        Returns Pandas DataFrame of LiDaR data.

        Parameters
        ----------
        sample_size : float, optional
            percent of the data to be returned. The default is 1.

        Returns
        -------
        lidar_df : pd.DataFrame
            lidar data.

        """

        # get all LiDaR data in the data directory
        lidar_data = glob.glob(str(self.path_lidar)+"*.bin")

        lidar_df = pd.DataFrame()
        for frame, file in enumerate(lidar_data):
            lidar_dict = self.__load_lidar_helper(file, sample_size)
            temp_df = pd.DataFrame.from_dict(lidar_dict)
            temp_df['frame'] = frame
            lidar_df = lidar_df.append(temp_df)

        self.lidar_data = lidar_df
        return lidar_df

    def load_imu(self):
        """
        Returns Pandas DataFrame of IMU data.

        Returns
        -------
        imu_df : pd.DataFrame
            IMU data.

        """
        # load the timestamps
        time_file = open(str(self.path_imu)+ "../timestamps.txt", "r")
        content = time_file.read()
        timestamps = content.split("\n")
        time_file.close()

        if timestamps[-1] == "":
            timestamps = timestamps[:-1] # remove empty line

        # turn timestamps into datetime object
        timestamps_obj = list()
        for t in timestamps:
            cut = len(t.split(".")[-1]) - 6
            if cut > 0:
                date_time_obj = datetime.strptime(t[:-cut], '%Y-%m-%d %H:%M:%S.%f')
            else:
                date_time_obj = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')

            timestamps_obj.append(date_time_obj)

        # calculate the time between each measurement
        differences = list()
        differences.append(0)
        for ii, obj in enumerate(timestamps_obj):
            if ii+1 == len(timestamps_obj):
                break
            differences.append((timestamps_obj[ii+1] - obj).total_seconds())

        # get all IMU data in the data directory
        imu_information = glob.glob(str(self.path_imu)+"*.txt")

        imu_df = pd.DataFrame()
        for frame, file in enumerate(imu_information):
            temp = pd.read_csv(str(file), header=None,
                               delimiter=r"\s+", names=self.imu_columns)
            temp['frame'] = frame
            temp['time_delta'] = differences[frame]
            imu_df = imu_df.append(temp)

        self.imu_data = imu_df
        return imu_df

    def get_params(self):
        """
        Returns the object variables.

        Returns
        -------
        dict
            self.

        """

        return vars(self)

    def __load_lidar_helper(self, file, sample_size):
        """
        Helper function to load the lidar data

        Parameters
        ----------
        file : str
            current file.
        sample_size : float
            percentage of the lidar points to be loaded.

        Returns
        -------
        dict
            lidar coordinates. point cloud.

        """

        pointcloud = np.fromfile(str(file), dtype=np.float32,
                                 count=-1).reshape([-1, 4])
        if sample_size != 1:
            pointcloud = pointcloud[np.random.choice(pointcloud.shape[0], int(
                pointcloud.shape[0] * sample_size), replace=False), :]

        x = pointcloud[:, 0]  # x position of point
        y = pointcloud[:, 1]  # y position of point
        z = pointcloud[:, 2]  # z position of point

        r = pointcloud[:, 3]  # reflectance value of point
        d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

        return {"x": x, "y": y, "z": z, "r": r, "d": d}
