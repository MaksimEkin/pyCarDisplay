import PySimpleGUI as sg
import pandas as pd
from PIL import Image
import io
from io import StringIO

class Display():

    # need to pass a frame dictionary that contains dictionaries of image paths and detected image lists

    def __init__(self, speed: int, total_frames: int):
        self.cropped_img_displayed = 5
        self.close = sg.WIN_CLOSED
        self.speed = speed
        self.total_frames = total_frames
        self.verbose = False
        # Create the window
        self.theme = sg.theme("DarkGrey1")
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

            if num < self.cropped_img_displayed:
                self.update_window("IMG" + str(num + 1), self.format_pil_img(detected_image), detected_image.size)


    def speed_update(self, imu_data, kalman_imu_data):
        if self.verbose:
            print("Examine imu=", imu_data['data'][0])
            print("Examine Kalman=", kalman_imu_data['data'][0])

    def grid_update(self, imu_data, kalman_imu_data):

        for row,(key, df) in enumerate(imu_data.items()):
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                if key == "noise":
                    df = df.T
            for col, entry in enumerate(list(df.iloc[0].values)):
                self.update_window(str(row) + "," + str(col), round(entry,2))


    def reset_depth_images(self, cropped_depth_images):
        for num in range(self.cropped_img_displayed):
            self.update_window("IMG" + str(num + 1), "")

    def define_layout(self):

        headings = ['lat', 'lon', 'alt', 'roll',
        'pitch', 'yaw', 'vn', 've','vf', 'vl', 'vu', 'ax', 'ay', 'az',
        'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'pos_accuracy',
        'vel_accuracy', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode']

        header =  [[sg.Text(h, size=(6,1)) for h in headings]]
        input_rows = [[sg.Input(size=(6,1), pad=(8,0), key=str(row)+","+str(col)) for col in range(len(headings))] for row in range(4)]

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
            ],
        ] + header + input_rows

    def end(self):
        self.window.close()

    def play(self, annotated_image: Image, cropped_depth_images: list, imu_data: pd.DataFrame,
             kalman_imu_data: pd.DataFrame, frame: int, verbose:bool, ):


        cropped_depth_images = ['heh', 'ehh']

        self.verbose = verbose
        if self.verbose:
            print(type(annotated_image))

        # check if pause or play were clicked or if window closed
        """May need to relocate this"""
        event, values = self.window.read(timeout=1)

        # Reset objects no longer detected in frame
        self.reset_depth_images(cropped_depth_images)

        # update main display_api with detected objects
        #annotated_image.show()
        with io.BytesIO() as output:
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
        self.grid_update(imu_data, kalman_imu_data)
        #self.update_window("0,3", str(frame))

        self.progress_bar.UpdateBar(frame + 1)
