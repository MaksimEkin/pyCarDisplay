"""

"""
import os

from .utils.display_api import Display
from .utils.kitti_data_loader_api import DataLoader

from .sensors.image_processing_api import ImageProcessing
from .sensors.imu_api import IMU
from .sensors.kalman_filter_api import KalmanFilter

from .detection.object_detection_api import ObjectDetection
from .detection.depth_detection_api import DepthDetection


class Car():
    def __init__(self, 
                 # Required
                 car_images_path:str, 
                 imu_sensor_path:str,
                 lidar_sensor_path:str,
                 object_detection_model_path:str,
                 depth_detection_model_path:str,
                 verbose:bool,
                 
                 # Object detection hyper-parameters
                 img_resize_size=(300, 300),
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225],
                 
                 # IMU parameters
                 R_covariance=0.1,
                 
                 # Other
                 random_state=42
                 ):
        """"""
        self.car_images_path = car_images_path
        self.imu_sensor_path = imu_sensor_path
        self.verbose = verbose
        
        # Load the object detection API
        self.obj_detection_api = ObjectDetection(object_detection_model_path, 
                                                 verbose,
                                                 img_resize_size,
                                                 norm_mean,
                                                 norm_std)
        
        # Load the Kitti IMU data
        imu_df = DataLoader(imu_sensor_path, lidar_sensor_path).load_imu()
        self.imu_sensor = IMU(imu_df,
                              verbose,
                              R_covariance,
                              random_state
            )
        
        # Image processing API
        self.img_processing_api = ImageProcessing(self.car_images_path)
        self.path_to_all_images = self.img_processing_api.process_images_path()
        
        # Display API
        self.display_api = Display()


    def run(self):
        """"""
        
        # TODO: process frames in a for loop and display them
        
        return -1