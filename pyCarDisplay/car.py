"""

"""
import os

from .utils.display_api import Display
from .utils.kitti_data_loader_api import DataLoader

from .sensors.image_processing_api import ImageProcessing
from .sensors.imu_api import IMU
from .sensors.kalman_filter_api import KalmanFilter

from .detection.object_detection_api import ObjectDetection
#from .detection.depth_detection_api import DepthDetection

from PIL import Image
import sys

class Car():
    def __init__(self,
                 # Required
                 car_images_path:str,
                 imu_sensor_path:str,
                 lidar_sensor_path:str,
                 object_detection_model_path:str,
                 depth_detection_model_path:str,
                 

                 # Object detection hyper-parameters
                 img_resize_size=(300, 300),
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225],

                 # IMU parameters
                 R_covariance=0.1,
                 add_noise=True,
                 IMU_names=None,

                 # Display API Required
                 gui_speed=1,
                 
                 # Image processing API
                 image_extension="png",

                 # Other
                 random_state=42,
                 verbose=False
                 ):
        """ Initialize car object data"""

        self.car_images_path = car_images_path
        self.imu_sensor_path = imu_sensor_path
        self.verbose = verbose
        self.add_noise = add_noise
        self.IMU_names = IMU_names
        self.gui_speed=gui_speed
        
        if self.verbose:
            print("Car configurations:\n" + \
            "car_images_path = " + str(car_images_path) +"\n" + \
            "imu_sensor_path = " + str(imu_sensor_path) +"\n" + \
            "lidar_sensor_path = " + str(lidar_sensor_path) + "\n" + \
            "object_detection_model_path = " + str(object_detection_model_path) +"\n"+ \
            "depth_detection_model_path = " + str(depth_detection_model_path) +"\n"+ \
            "img_resize_size = " + str(img_resize_size) + "\n"+\
            "norm_mean = " + str(norm_mean) +"\n"+\
            "norm_std = " + str(norm_std) +"\n"+\
            "R_covariance = " + str(R_covariance) + "\n"+\
            "add_noise = " + str(add_noise) +"\n"+\
            "IMU_names = " + str(IMU_names) +"\n"+\
            "gui_speed = " + str(gui_speed) + "\n"+\
            "random_state = " + str(random_state) +"\n"+\
            "image_extension = " + str(image_extension) +"\n"+\
            "verbose = " + str(verbose))

        # Load the object detection API
        self.obj_detection_api = ObjectDetection(object_detection_model_path,
                                                 verbose,
                                                 img_resize_size,
                                                 norm_mean,
                                                 norm_std)

        # Load the depth detection API
        #self.depth_detection_api = DepthDetection(depth_detection_model_path,
        #                                          verbose,
        #                                          img_resize_size,
        #                                          norm_mean,
        #                                          norm_std)

        #self.ml_synchronize_api = MLDataSynch()

        # Load the Kitti IMU data
        if self.verbose:
            print("Loading the IMU data...")
        imu_df = DataLoader(path_imu=imu_sensor_path, path_lidar=lidar_sensor_path).load_imu()
        
        if self.verbose:
            print(imu_df.info())
        if len(imu_df) == 0:
            sys.exit("Failed to load the IMU data.")
        
        self.imu_sensor = IMU(imu_df,
                              verbose,
                              R_covariance,
                              random_state
            )

        # Image processing API
        self.img_processing_api = ImageProcessing(
            car_images_path=self.car_images_path, 
            image_extension=image_extension)
        self.path_to_all_images = self.img_processing_api.all_images_path
        if self.verbose:
            print("Total image frames loaded:" + str(len(self.path_to_all_images)))
        
        self.total_frames = len(self.path_to_all_images)
        assert self.total_frames == len(imu_df)

        # Display API
        self.display_api = Display(self.gui_speed, self.total_frames)


    def set_frame(self, frame:int):
        """"""
        self.imu_sensor.set_frame(frame)



    def run(self):
        """Iterates the taken images and one by one performs object detection and depth detection on the objects
            renders the imgaes taken on a display with the depth heatmaps as subimages."""

        for curr_frame, curr_img_path in enumerate(self.path_to_all_images):

            # Simulate taking a picture (comes from files in a directory rather than camera)
            image = self.img_processing_api.take_picture(curr_img_path)


            # return data: {"annotated_image": PIL.Image,"box_info":{"text_size":list(),"box_location":list()} }
            detected_dictionary = self.obj_detection_api.detect(image)

            # return data: PIL.Image
            #depth_image, depth_information = self.depth_detection_api.detect(image)

            # return data: list
            cropped_depth_images = self.img_processing_api.synch_objDet_depDet_data(
                detected_dictionary["box_info"]["box_location"]#,
                #depth_image,
                #depth_information
            )

            # object_detected_image:PIL.Image, depth_images:list, depths:list, imu_data:pd.DataFrame, kalman_imu_data:pd.DataFrame, frame:int
            self.display_api.play(
                detected_dictionary["annotated_image"],
                #cropped_depth_images,
                self.imu_sensor.read_sensor(add_noise=self.add_noise, IMU_names=self.IMU_names),
                {},
                curr_frame
            )

        self.display_api.end()
        return -1
