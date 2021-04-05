"""
See config.py in the root directory to specify the path to the dataset.
"""

import pandas as pd
import numpy as np
import glob


class DataLoader(object):
    def __init__(self, path_lidar="", path_imu=""):
        '''
        If path is empty string, path in the config file is used for the dataset.
        '''
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
        '''
        Returns Pandas DataFrame of LiDaR data.
        
        Paramters
            sample_size: float, percent of the data to be returned.
        '''
        
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
        '''
        Returns Pandas DataFrame of IMU data.
        '''
        
        # get all IMU data in the data directory
        lidar_data = glob.glob(str(self.path_imu)+"*.txt")  
        
        imu_df = pd.DataFrame()
        for frame, file in enumerate(lidar_data):
            temp = pd.read_csv(str(file), header=None, delimiter=r"\s+", names=self.imu_columns)
            temp['frame'] = frame
            imu_df = imu_df.append(temp)
            
        self.imu_data = imu_df
        return imu_df
    
    def get_params(self):
        return vars(self)

    def __load_lidar_helper(self, file, sample_size):

        pointcloud = np.fromfile(str(file), dtype=np.float32,
                             count=-1).reshape([-1, 4])
        if sample_size != 1:
            pointcloud = pointcloud[np.random.choice(pointcloud.shape[0], int(pointcloud.shape[0] * sample_size), replace=False), :]
            
        x = pointcloud[:, 0]  # x position of point
        y = pointcloud[:, 1]  # y position of point
        z = pointcloud[:, 2]  # z position of point

        r = pointcloud[:, 3]  # reflectance value of point
        d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

        return {"x":x, "y":y, "z":z, "r":r, "d":d}