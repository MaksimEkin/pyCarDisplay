"""
All of the Car operations are controlled from this file.
"""

from .utils.display_api import Display
from .utils.kitti_data_loader_api import DataLoader

from .sensors.image_processing_api import ImageProcessing
from .sensors.imu_api import IMU
from .sensors.kalman_filter_api import KalmanFilter

from .detection.object_detection_api import ObjectDetection
from .detection.depth_detection_api import DepthDetection
from PIL import Image
import sys
import numpy as np
from termcolor import colored


class Car():

    def __init__(self, car_images_path: str, imu_sensor_path="", lidar_sensor_path="",
                 object_detection_model_path="", depth_detection_model_path="",
                 depth_detection_model_type="large", optimize=True, img_resize_size=(300, 300),
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225],
                 R_covariance=0.1, add_noise=True, IMU_name=None, gui_speed=1,
                 theme="DarkGrey1", image_extension="png", random_state=42,
                 verbose=False, device="cpu"):
        """
        Initilize the Car class.

        Parameters
        ----------
        car_images_path : str
            Path to the Kitti dataset image frames.
        imu_sensor_path : str, optional
            Path to the Kitti dataset IMU sensor data. The default is "".
            If not provided, simulation will not work on the IMU data.
        lidar_sensor_path : str, optional
            Path to the Kitti dataset Lidar sensor data. The default is "".
            If not provided, simulation will not work on the Lidar data.
        object_detection_model_path : str, optional
            Path to the trained PyTorch object detection model. The default is "".
            If not provided, object detection is not performed.
        depth_detection_model_path : str, optional
            Path to the trained PyTorch depth detection model. The default is "".
        depth_detection_model_type : str, optional
            If 'large', model uses a larger trained model but it will be slower. The default is "large".
        optimize : bool, optional
            If True, images are pre-processed for better depth detection. The default is True.
        img_resize_size : tuple, optional
            Size of the image when performing object detection. The default is (300, 300).
        norm_mean : list, optional
            Object detection hyper-parameter. The default is [0.485, 0.456, 0.406].
        norm_std : list, optional
            Object detection hyper-parameter. The default is [0.229, 0.224, 0.225].
        R_covariance : float, optional
            Error covariance for IMU data. The default is 0.1.
        add_noise : bool, optional
            If True, random uniform noise added to the IMU data. The default is True.
        gui_speed : int, optional
            Time between each frame in GUI. The default is 1.
        theme : str, optional
            GUI color theme. The default is "DarkGrey1".
        image_extension : str, optional
            Kitti dataset image extension. The default is "png".
            Used when reading the images from the path.
        random_state : int, optional
            random seed. The default is 42.
        verbose : bool, optional
            Flag to print information. The default is False.
        device : str, optional
            If 'gpu' passed, ML modules will use the GPU. The default is "cpu".
            CUDA device must be available.

        Returns
        -------
        None.

        """

        self.car_images_path = car_images_path
        self.imu_sensor_path = imu_sensor_path
        self.verbose = verbose
        self.add_noise = add_noise
        self.gui_speed = gui_speed

        # Print the settings
        if self.verbose:
            print(colored("Car configurations:\n" +
                          "car_images_path = " + str(car_images_path) + "\n" +
                          "imu_sensor_path = " + str(imu_sensor_path) + "\n" +
                          "lidar_sensor_path = " + str(lidar_sensor_path) + "\n" +
                          "object_detection_model_path = " + str(object_detection_model_path) + "\n" +
                          "depth_detection_model_path = " + str(depth_detection_model_path) + "\n" +
                          "img_resize_size = " + str(img_resize_size) + "\n" +
                          "norm_mean = " + str(norm_mean) + "\n" +
                          "norm_std = " + str(norm_std) + "\n" +
                          "R_covariance = " + str(R_covariance) + "\n" +
                          "add_noise = " + str(add_noise) + "\n" +
                          "gui_speed = " + str(gui_speed) + "\n" +
                          "random_state = " + str(random_state) + "\n" +
                          "image_extension = " + str(image_extension) + "\n" +
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

        # Load the Kitti IMU data
        if imu_sensor_path != "" or lidar_sensor_path != "":
            if self.verbose:
                print(colored("Loading the IMU data...", "blue"))
            imu_df = DataLoader(path_imu=imu_sensor_path,
                                path_lidar=lidar_sensor_path).load_imu()

            if self.verbose:
                print(imu_df.info())
            if len(imu_df) == 0:
                sys.exit("Failed to load the IMU data.")

            self.imu_sensor = IMU(imu_df,
                                  verbose,
                                  R_covariance,
                                  random_state)

            # Kalman Filter API -- depends on IMU sensor data
            self.kalman_filters = [KalmanFilter() for x in range(len(list(self.imu_sensor.imu_data.columns)))]
            print(self.kalman_filters)
            self.kalman_data_points =  {'data':[0 for x in range(len(list(self.imu_sensor.imu_data.columns)))]}

        else:
            self.imu_sensor = None

        # Image processing API
        self.img_processing_api = ImageProcessing(
            car_images_path=self.car_images_path,
            image_extension=image_extension)
        self.path_to_all_images = self.img_processing_api.all_images_path
        if self.verbose:
            print(colored("Total image frames loaded:" +
                          str(len(self.path_to_all_images)), "yellow"))

        # verify the total frames
        self.total_frames = len(self.path_to_all_images)
        if self.total_frames <= 0:
            sys.exit(
                "Number of frames must be more than 0. Detected 0 or less frames of images.")

        if imu_sensor_path != "":
            assert self.total_frames == len(imu_df)

        # Display API
        self.display_api = Display(self.gui_speed, self.total_frames, theme)

    def set_frame(self, frame: int):
        """
        Manually sets the frame.

        Parameters
        ----------
        frame : int
            Target frame.

        Returns
        -------
        None.

        """
        if frame < 0 or frame >= self.total_frames:
            sys.exit("Invalid frame number requested.")
        self.imu_sensor.set_frame(frame)

    def run(self, verbose=None):
        """
        Iterates the taken images and one by one performs object detection and depth detection on the objects
        renders the imgaes taken on a display with the depth heatmaps as subimages.

        Parameters
        ----------
        verbose : Bool
            Flag to print more information at runtime.

        Returns
        -------
        None. Displays GUI.

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
                detected_dictionary = {"annotated_image": image, "box_info": {
                    "text_size": [], "box_location": []}, "detected": False}

            # detect depth
            if self.depth_detection_api != None:
                depth_image = self.depth_detection_api.run(
                    self.verbose, image, True)
                depth_min = depth_image.min()
                depth_max = depth_image.max()
                max_val = (2 * (81)) - 1
                out = max_val * (depth_image - depth_min) / \
                    (depth_max - depth_min)
                #pil_depth_image = Image.fromarray(np.uint8(cm.gist_earth(np.log2(out))*255), 'RGBA')
                pil_depth_image = Image.fromarray(np.log2(out), 'RGBA')

            else:
                pil_depth_image = image

            # read IMU sensor data
            if self.imu_sensor != None:
                curr_imu_data = self.imu_sensor.read_sensor(
                    add_noise=self.add_noise, name=None)

                print(curr_imu_data['data'])

                for i, col in enumerate(list(curr_imu_data['data'].columns)):
                    #print("col", col)
                    #print("curr_imu_data['data'][col].values", curr_imu_data['data'][col].values[0])
                    predict = self.kalman_filters[i].Predict(curr_imu_data['data'][col].values[0], #SESNSOR READ
                                                             self.kalman_data_points['data'][i], #PREVIOUS DATA POINT
                                                             1) # DELTA TIME

                    #set next data points based on the prediction, senssor and covariance
                    self.kalman_data_points['data'][i] = self.kalman_filters[i].Update(curr_imu_data["data"][col].values[0],
                                                                                        self.imu_sensor.R_covariance,
                                                                                        predict)

            else:
                curr_imu_data = {}


            # Display the current frame
            self.display_api.play(
                detected_dictionary["annotated_image"],
                # cropped_depth_images,
                pil_depth_image,
                curr_imu_data,
                self.kalman_data_points, #{'data': [45.456]}, # Kalman data
                curr_frame,
                self.verbose
            )

        self.display_api.end()
        return -1
