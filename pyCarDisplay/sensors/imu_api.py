"""

"""
import numpy as np
import pandas as pd
import warnings
import sys

class IMU:
    def __init__(self, data, verbose=True, R_covariance=0.1, random_state=42):
        assert type(data) is pd.DataFrame, "data should be type Pandas DataFrame"
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
        return self.R_covariance
            
    def get_avail_data(self):
        return list(self.imu_data.columns) 
        
    def get_params(self):
        return vars(self)
    
    def read_sensor(self, name=None, add_noise=True, advance_frame=True):
        
        if self.frame > self.num_frames:
            if self.verbose:
                warnings.warn("Reached at the end of available number of frames. Reverting to frame 0. Number of frames: " + str(self.num_frames))
            self.frame = 0
        
        noise = 0
        
        data = None
        if name is None:
            
            if add_noise:
                num_columns = len(self.params_avail) - 1 # -1 for the frame column
                noise = np.random.normal(0, self.R_covariance, num_columns)
                data = self.imu_data[self.imu_data.frame == self.frame].loc[:, self.imu_data.columns != "frame"] + noise
            else:
                data = self.imu_data[self.imu_data.frame == self.frame].loc[:, self.imu_data.columns != "frame"]
        else:
            self.__check_param(name)
            if add_noise:
                noise = np.random.normal(0, self.R_covariance, 1)
            data =  self.imu_data[self.imu_data.frame == self.frame][str(name)].values + noise
        
        if advance_frame:
            self.frame += 1
        return {"data": data, "noise": noise, "true": data-noise}
        
    def get_column(self, name:str):
        self.__check_param(name)
        return np.array(self.imu_data[str(name)])
    
    def set_frame(self, frame:int):
        if frame > self.num_frames or frame < 0:
            sys.exit("Frame has to be a positive intiger, and has to be less than the total number of available frames. Number of frames: " + str(self.num_frames))
        self.frame = frame
    
    def __check_param(self, name):
        assert name in self.params_avail, "Parameter is not available in IMU. Parameters: " + ", ".join(self.params_avail)