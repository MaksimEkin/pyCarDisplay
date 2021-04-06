import PySimpleGUI as sg
import pandas as pd
from PIL import Image
import io
from io import StringIO

class Display():

    # need to pass a frame dictionary that contains dictionaries of image paths and detected image lists
<<<<<<< HEAD
    def __init__(self, speed: int, total_frames: int):
        self.cropped_img_displayed = 5
        self.close = sg.WIN_CLOSED
        self.speed = speed
        self.total_frames = total_frames

=======
    def __init__(o, speed: int, total_frames: int):
        o.cropped_img_displayed = 5
        o.close = sg.WIN_CLOSED
        o.speed = speed
        o.total_frames = total_frames
        o.verbose = False
>>>>>>> ba0a8dc4e080f0daed0d03dcbb3e6f0fe71ef578
        # Create the window
        self.window = sg.Window("Autonomous Vehicle Object & Distance Detection", self.define_layout())
        self.progress_bar = self.window['progressbar']

    def img(self, path, key):
        return sg.Image(path, key=key)

    def update_window(self, key, data1, data2=''):
        if data2:
            self.window.FindElement(key).Update(data=data1, size=data2)
        else:
            self.window.FindElement(key).Update(data1)

    def depth_images_update(self, cropped_depth_images):
        for num, detected_image in enumerate(cropped_depth_images):
<<<<<<< HEAD
            if num < self.cropped_img_displayed:
                self.update_window("IMG" + str(num + 1), self.format_pil_img(detected_image), detected_image.size)
=======
            if num < o.cropped_img_displayed:
                o.update_window("IMG" + str(num + 1), o.format_pil_img(detected_image), detected_image.size)


    def speed_update(o, imu_data, kalman_imu_data):
        if o.verbose:
            print("Examine imu=", imu_data['data'][0])
            print("Examine Kalman=", kalman_imu_data['data'][0])
>>>>>>> ba0a8dc4e080f0daed0d03dcbb3e6f0fe71ef578

    def speed_update(self, imu_data, kalman_imu_data):
        gold_speed = "True Speed: " + str(imu_data['data'][0])
<<<<<<< HEAD
        self.window.FindElement("speed").Update(gold_speed)
=======
        
        if o.verbose:
            print("gold_speed", gold_speed)
        o.window.FindElement("speed").Update(gold_speed)
>>>>>>> ba0a8dc4e080f0daed0d03dcbb3e6f0fe71ef578

        kalman_speed = "Kalman speed: " + str(kalman_imu_data['data'][0])
        self.update_window("kspeed", kalman_speed)

    def reset_depth_images(self, cropped_depth_images):
        for num in range(self.cropped_img_displayed):
            self.update_window("IMG" + str(num + 1), "")

    def define_layout(self):
        return [
            [sg.ProgressBar(self.total_frames, orientation='h', size=(50, 5), key='progressbar')],
            [sg.Text("Frame: 1", size=(50, 1), key="frame")],
            [sg.Text("True Speed:" + " " * 30 + str(self.speed), key="speed")],
            [sg.Text("Kalman speed:"+ " " * 20 + str(self.speed), key="kspeed")],
            [self.img("", "IMG")],
            [
                self.img("", "IMG1"),
                self.img("", "IMG2"),
                self.img("", "IMG3"),
                self.img("", "IMG4"),
                self.img("", "IMG5")
            ]
        ]

    def end(self):
        self.window.close()

<<<<<<< HEAD
    def play(self, annotated_image: Image, cropped_depth_images: list, imu_data: pd.DataFrame,
             kalman_imu_data: pd.DataFrame, frame: int):
=======
    def play(o, annotated_image: Image, cropped_depth_images: list, imu_data: pd.DataFrame,
             kalman_imu_data: pd.DataFrame, frame: int, verbose:bool):
        
        
>>>>>>> ba0a8dc4e080f0daed0d03dcbb3e6f0fe71ef578
        cropped_depth_images = ['heh', 'ehh']
        
        o.verbose = verbose
        if o.verbose:
            print(type(annotated_image))

        # check if pause or play were clicked or if window closed
        """May need to relocate this"""
        event, values = self.window.read(timeout=1)

        # Reset objects no longer detected in frame
        self.reset_depth_images(cropped_depth_images)

        # update main display_api with detected objects
        #annotated_image.show()
        with iself.BytesIO() as output:
            annotated_image.save(output, format="PNG")
            contents = output.getvalue()
        self.update_window("IMG", contents, annotated_image.size)

        # update up to cropped_img_displayed number of the depth images of detected objects
        ##for num, detected_image in enumerate(cropped_depth_images):
            ##if num < self.cropped_img_displayed:
                #self.update_window("IMG" + str(num + 1), detected_image, detected_image.size)

        # current picture frame
        self.update_window("frame", "Frame: " + str(frame))

        # current Speed and Kalman speed updated with api data
        self.speed_update(imu_data, kalman_imu_data)
        self.progress_bar.UpdateBar(frame + 1)
