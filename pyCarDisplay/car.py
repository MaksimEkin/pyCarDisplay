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
from PIL import Image
import sys
from matplotlib import cm
import numpy as np
from termcolor import colored

class Car():
    """

    """
    def __init__(self,
                 # Required Path
                 car_images_path:str,
                 
                 # Other paths
                 imu_sensor_path="",
                 lidar_sensor_path="",
                 object_detection_model_path="",
                 depth_detection_model_path="",
                 
                 # Depth detection
                 depth_detection_model_type="large",
                 optimize=True,

                 # Object detection hyper-parameters
                 img_resize_size=(300, 300),
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225],

                 # IMU parameters
                 R_covariance=0.1,
                 add_noise=True,
                 IMU_name=None,

                 # Display API settings
                 gui_speed=1,
                 theme = "DarkGrey1",
                 
                 # Image processing API
                 image_extension="png",

                 # Other
                 random_state=42,
                 verbose=False,
                 device="cpu"
                 ):
        """ 
        Initialize car object data
        """

        self.car_images_path = car_images_path
        self.imu_sensor_path = imu_sensor_path
        self.verbose = verbose
        self.add_noise = add_noise
        self.IMU_name = IMU_name
        self.gui_speed=gui_speed

        if self.verbose:
            print(colored("Car configurations:\n" + \
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
            "IMU_name = " + str(IMU_name) +"\n"+\
            "gui_speed = " + str(gui_speed) + "\n"+\
            "random_state = " + str(random_state) +"\n"+\
            "image_extension = " + str(image_extension) +"\n"+\
            "verbose = " + str(verbose), "cyan"))

        # Load the object detection API
        if object_detection_model_path != "":
            self.obj_detection_api = ObjectDetection(object_detection_model_path,
                                                     verbose,
                                                     img_resize_size,
                                                     norm_mean,
                                                     norm_std,
                                                     device
                                                    )
        else:
            self.obj_detection_api = None

        # Load the depth detection API
        if depth_detection_model_path != "":
            self.depth_detection_api = DepthDetection(self.verbose,
                                                      depth_detection_model_path,
                                                      depth_detection_model_type,
                                                      device=device 
                                                     )
        else:
            self.depth_detection_api = None

        # Initialize a Kalman Filter (P, H, F for speed sensor1 and 2, Q_covariance's mean)
        #self.kalman_filter = KalmanFilter(P=1, H=1, F=1, Q_covariance=0.1)

        # Load the Kitti IMU data
        if imu_sensor_path != "" or lidar_sensor_path != "":
            if self.verbose:
                print(colored("Loading the IMU data...", "blue"))
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
        else:
            self.imu_sensor = None

        # Image processing API
        self.img_processing_api = ImageProcessing(
            car_images_path=self.car_images_path,
            image_extension=image_extension)
        self.path_to_all_images = self.img_processing_api.all_images_path
        if self.verbose:
            print(colored("Total image frames loaded:" + str(len(self.path_to_all_images)), "yellow"))

        self.total_frames = len(self.path_to_all_images)
        if self.total_frames <= 0:
            sys.exit("Number of frames must be more than 0.")
        
        if imu_sensor_path != "":
            assert self.total_frames == len(imu_df)

        # Display API
        self.display_api = Display(self.gui_speed, self.total_frames, theme)


    def set_frame(self, frame:int):
        """
        Sets the car's current frome to the most recent image reading
        
        Parameters
        ----------
        frame : int
            picture frame number of current image

        Returns
        -------
        None
        """
        self.imu_sensor.set_frame(frame)



    def run(self, verbose=None):
        """
        Iterates the taken images and one by one performs object detection and depth detection on the objects
        renders the imgaes taken on a display with the depth heatmaps as subimages.

        Parameters
        ----------
        verbose : Bool
            Flag to print more information at runtime

        Returns
        -------
        None
        """

        # if verbose is being changed
        if verbose != None:
            self.verbose = verbose

        for curr_frame, curr_img_path in enumerate(self.path_to_all_images):

            # Simulate taking a picture (comes from files in a directory rather than camera)
            image = self.img_processing_api.take_picture(curr_frame)

            # detect object
            if self.obj_detection_api != None:
                detected_dictionary = self.obj_detection_api.detect(image)
            else:
                detected_dictionary = {"annotated_image":image, "box_info":{"text_size":[],"box_location":[]}, "detected":False}

            # detect depth
            if self.depth_detection_api != None:
                depth_image = self.depth_detection_api.run(self.verbose, image, True)
                depth_min = depth_image.min()
                depth_max = depth_image.max()
                max_val = (2 * (81)) - 1
                out = max_val * (depth_image - depth_min) / (depth_max - depth_min)
                #pil_depth_image = Image.fromarray(np.uint8(cm.gist_earth(np.log2(out))*255), 'RGBA')
                pil_depth_image = Image.fromarray(np.log2(out), 'RGBA')
                
            else:
                pil_depth_image = image
                
            
            # read IMU sensor data
            if self.imu_sensor != None:
                curr_imu_data = self.imu_sensor.read_sensor(add_noise=self.add_noise, name=self.IMU_name)
            else:
                curr_imu_data = {}
                
            # Display the current frame
            self.display_api.play(
                detected_dictionary["annotated_image"],
                #cropped_depth_images,
                pil_depth_image,
                curr_imu_data,
                {'data':[45.456]},
                curr_frame,
                self.verbose
            )

        self.display_api.end()
        return -1
