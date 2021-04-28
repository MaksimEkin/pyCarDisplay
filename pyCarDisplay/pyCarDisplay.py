"""
pyCarDisplay is a autonomous vehicle simulator. It is designed to simulate driving car
based on the datasets provided in Kitti (http://www.cvlibs.net/datasets/kitti/).
pyCarDisplay can show the image frames in a Kitti dataset on a GUI, perform object
detection given a trained model, perform depth detection given a trained model,
simulate IMU sensor given IMU data, and use Kalman filter.

pyCarDisplay class is used as an API wrapper to the rest of the modules, i.e.
the simulation is invoked from this file.
"""
import os
from termcolor import colored


class CarDisplay():
    """
    The main API class of the pyCarDisplay.
    """

    def __init__(self, **parameters):
        """
        Initilize the CarDisplay class. This class passes the below
        parameters to the Car class.

        The initlization also sets up the environment variables.

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
         dpi : int, optional
             Dots per inch. The default is 100.
         alpha : double, optional
             The alpha variable. The default is 0.6.
         img_resize_size : tuple, optional
             Size of the image when performing object detection. The default is (300, 300).
         pixel_sizes : list, optional
             The sizes of the pixels. The default is [1242, 375].
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
        track_n_frames : int, optional
            How many frames to track in the Kalman plot. The default is 10.
        plot_column : str, optional
            Which IMU column to plot in the Kalman plot. The default is "af".

        Returns
        -------
        None.

        """

        # Set environment variables
        if "device" in list(parameters.keys()):
            os.environ["PYCARDISPLAY_DEVICE"] = parameters["device"]
        else:
            os.environ["PYCARDISPLAY_DEVICE"] = "cpu"

        # show the envrionment variable
        if "verbose" in list(parameters.keys()):
            if parameters["verbose"]:
                print(
                    colored("Environment variable is set. PYCARDISPLAY_DEVICE =" +
                            str(os.environ["PYCARDISPLAY_DEVICE"]),
                            "yellow"))

        from .car import Car
        self.car = Car(**parameters)

    def start(self, **parameters):
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

        self.car.run(**parameters)

    def get_params(self):
        """
        Can be used to get the object variables.

        Returns
        -------
        self
            The object variables in dict format.

        """
        return vars(self.car)

    def set_params(self, verbose=True, **parameters):
        """
        Can be used to change the Car settings.


        Parameters
        ----------
        verbose : bool, optional
            Used the show warning. The default is True.
        **parameters : dict
            all the object variables accepted by the car class.

        Returns
        -------
        None.

        """
        if verbose:
            print("Changing the vehicle parameters does not reload the\
                  data and ML models. Please re-start the program to change\
                  these, or use this function to modify the hyper-parameters.")

        for variable, value in parameters.items():
            setattr(self.car, variable, value)
