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

class Car():
    """

    """
    def __init__(self,
                 # Required
                 car_images_path:str,
                 imu_sensor_path:str,
                 lidar_sensor_path:str,
                 object_detection_model_path:str,
                 depth_detection_model_path:str,
                 
                 # Depth detection
                 depth_detection_model_type="large",

                 # Object detection hyper-parameters
                 img_resize_size=(300, 300),
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225],

                 # IMU parameters
                 R_covariance=0.1,
                 add_noise=True,
                 IMU_name=None,

                 # Display API Required
                 gui_speed=1,
                 theme = "DarkGrey1",
                 

                 # Image processing API
                 image_extension="png",

                 # Other
                 random_state=42,
                 verbose=False,
                 optimize=True,
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
            "IMU_name = " + str(IMU_name) +"\n"+\
            "gui_speed = " + str(gui_speed) + "\n"+\
            "random_state = " + str(random_state) +"\n"+\
            "image_extension = " + str(image_extension) +"\n"+\
            "verbose = " + str(verbose))

        # Load the object detection API
        self.obj_detection_api = ObjectDetection(object_detection_model_path,
                                                 verbose,
                                                 img_resize_size,
                                                 norm_mean,
                                                 norm_std,
                                                 device
                                                )

        # Load the depth detection API
        self.depth_detection_api = DepthDetection(self.verbose,
                                                  depth_detection_model_path,
                                                  depth_detection_model_type,
                                                  device=device 
                                                 )

        # Initialize a Kalman Filter (P, H, F for speed sensor1 and 2, Q_covariance's mean)
        self.kalman_filter = KalmanFilter(P=1, H=1, F=1, Q_covariance=0.1)

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

            if self.verbose:
                print("Frame:" + str(curr_frame))

            # Simulate taking a picture (comes from files in a directory rather than camera)
            image = self.img_processing_api.take_picture(curr_frame)
            if self.verbose:
                print(image)

            # return data: {"annotated_image": PIL.Image,"box_info":{"text_size":list(),"box_location":list()} }
            detected_dictionary = self.obj_detection_api.detect(image)
            if self.verbose:
                print(detected_dictionary)
                print("Object detected:" + str(len(detected_dictionary["box_info"]["text_size"])))

            # return data: PIL.Image
            depth_image = self.depth_detection_api.run(self.verbose, image, True)
            #print("depth_image=", depth_image)

            depth_min = depth_image.min()
            depth_max = depth_image.max()

            max_val = (2 * (81)) - 1
            out = max_val * (depth_image - depth_min) / (depth_max - depth_min)
            #pil_depth_image = Image.fromarray(np.uint8(cm.gist_earth(np.log2(out))*255), 'RGBA')
            pil_depth_image = Image.fromarray(np.log2(out), 'RGBA')



            # return data: list
            #cropped_depth_images = self.img_processing_api.synch_objDet_depDet_data(
            #    detected_dictionary["box_info"]["box_location"]#,
                #depth_image,
                #depth_information
            #)

            curr_imu_data = self.imu_sensor.read_sensor(add_noise=self.add_noise, name=self.IMU_name)
            if self.verbose:
                print(curr_imu_data["data"])

            # object_detected_image:PIL.Image, depth_images:list, depths:list, imu_data:pd.DataFrame, kalman_imu_data:pd.DataFrame, frame:int
            cropped_depth_images = []

            #Kalman filter from old code, needs to be adapted to this section
            """
            for time in range(0, run_time):
                if time == 0:
                    # First, set the initial fused speed data to the vehicle's true speed
                    speed_true[time] = vehicle.speed
                    speed_fused[time] = speed_true[time]
        
                else:
                    sensor_1_read = imu_sensor_1.read_sensor("ax", advance_frame=False)
                    sensor_2_read = imu_sensor_2.read_sensor("ax", advance_frame=False)

                    # Kalman: Predict Step: Based on the previous best estimation to predict the current speed
                    speed_predict = kalman_filter.Predict(sensor_1_read["data"], speed_fused[time-1], delta_time)

                    # Kalman: Update Step:
                    sensor_1_read_2 = imu_sensor_1.read_sensor("vf")
                    sensor_2_read_2 = imu_sensor_2.read_sensor("vf")
                    speed_fused[time] = kalman_filter.Update(sensor_1_read_2["data"], imu_sensor_1.get_R(), speed_predict)
                    speed_fused[time] = kalman_filter.Update(sensor_2_read_2["data"], imu_sensor_2.get_R(), speed_fused[time])

                    # Then, use Kalman Filter to fuse the data from different sensors from time 2 to run_time    
                    speed_true[time] = sensor_1_read_2["true"]
            """
            
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
