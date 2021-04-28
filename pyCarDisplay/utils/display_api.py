"""
Autonomous vehicle display application window
"""
import PySimpleGUI as sg
import pandas as pd
from PIL import Image
import io

class Display():
    """
    Creates and updates the application window created with PySimpleGui using the Autonomous vehicle information
    for images and environmental observations
    """

    def __init__(self, speed: int, total_frames: int, theme:str):
        """
        Initializes the display class

        Parameters
        ----------
        speed : int
            speed of the autonomous vehicle
        total_frames :
            number of saved images to iterate
        theme :
            color of the PySimpleGui window as defined in https://user-images.githubusercontent.com/46163555/70382042-796da500-1923-11ea-8432-80d08cd5f503.jpg
        """
        self.cropped_img_displayed = 5
        self.close = sg.WIN_CLOSED
        self.speed = speed
        self.total_frames = total_frames
        self.verbose = False
        # Create the window
        self.theme = sg.theme(theme)
        self.window = sg.Window("Autonomous Vehicle Object & Distance Detection", self.define_layout())
        self.progress_bar = self.window['progressbar']

    def img(self, path, key):
        """
        Simplification of PySimpleGUI function

        Parameters
        ----------
        path : str
            path to the pil image.
        key : str
            name to find the image in the application window.

        Returns
        -------
            image_element: Pysimple Gui image element

        """
        image_element = sg.Image(path, key=key)
        return image_element

    def update_window(self, key, data1, data2=''):
        """
        Changes the displyed element to show updated information

        Parameters
        ----------
        key : str
            locates the app element in the window to change
        data1 : str
            value to set the element with
        data2 : str

        Returns
        -------
        None
        """

        if data2:
            self.window.FindElement(key).Update(data=data1, size=data2)
        else:
            self.window.FindElement(key).Update(data1)

    def depth_images_update(self, cropped_depth_images):
        """
        Iterates the cropped image list and updates the display with the predefined limit on image count

        Parameters
        ----------
        cropped_depth_images : list
            list of PIL images cropped from the depth detection image

        Returns
        -------
        None
        """
        for num, detected_image in enumerate(cropped_depth_images):

            if num < self.cropped_img_displayed:
                self.update_window("IMG" + str(num + 1), self.format_pil_img(detected_image), detected_image.size)


    def speed_update(self, imu_data, kalman_imu_data):
        if self.verbose:
            print("Examine imu=", imu_data['data'][0])
            print("Examine Kalman=", kalman_imu_data['data'][0])

    def grid_update(self, imu_data, kalman_imu_data):
        """
        Sets the tabled grid with all of the updated Inertial Measurment Unit (IMU) Data and Kalman filter data

        Parameters
        ----------
        imu_data :
        kalman_imu_data :

        Returns
        -------
        None
        """

        for row,(key, df) in enumerate(imu_data.items()):
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                if key == "noise":
                    df = df.T

            for col, entry in enumerate(list(df.iloc[0].values)):
                if col < 30:
                    self.update_window(str(row) + "," + str(col), round(entry,2))

        for col, entry2 in enumerate(kalman_imu_data['data']):
            if col < 30:
                self.update_window(str(3) + "," + str(col), round(entry2,2))

    def reset_depth_images(self, cropped_depth_images):
        """
        Iterates all of the cropped image elements in the window and sets them to empty
        Parameters
        ----------
        cropped_depth_images : list

        Returns
        -------
        None
        """
        for num in range(self.cropped_img_displayed):
            self.update_window("IMG" + str(num + 1), "")

    def define_layout(self):
        """
        Sets the layout for the PySimpleGui application window

        Returns
        -------
        elements : list
            List of List contains window elements
        """

        headings = ['lat', 'lon', 'alt', 'roll',
        'pitch', 'yaw', 'vn', 've','vf', 'vl', 'vu', 'ax', 'ay', 'az',
        'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'pos_accuracy',
        'vel_accuracy', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode']

        row_names = ["data", "noise", "true", "Kalman"]

        header =  [[sg.Text(" ", size=(6,1))] + [sg.Text(h, size=(6,1), pad=(1,0)) for h in headings]]
        input_rows = [[sg.Text(row_names[row], size=(6,1))] + [sg.Input(size=(6,1), pad=(1,1), key=str(row)+","+str(col)) for col in range(len(headings))] for row in range(4)]

        elements =  [
            [sg.ProgressBar(self.total_frames, orientation='h', size=(50, 5), key='progressbar')],
            [sg.Text("Frame: 1", size=(50, 1), key="frame")],
            #[sg.Text("True Speed:" + " " * 30 + str(self.speed), key="speed")],
            #[sg.Text("Kalman speed:"+ " " * 20 + str(self.speed), key="kspeed")],
            [sg.Text("\t\t"), self.img("", "IMG")],
            [sg.Text("\t\t"), self.img("", "IMG1")],
            [sg.Text("\t\t"), self.img("", "IMG2"),
                self.img("", "IMG3"),
                self.img("", "IMG4"),
                self.img("", "IMG5")
            ],
        ] + header + input_rows

        return elements

    def end(self):
        """
        Closes the application windows using pysimplegui

        Returns
        -------
        None
        """
        self.window.close()

    def play(self, annotated_image: Image, cropped_depth_images: Image, imu_data: pd.DataFrame,
             kalman_imu_data: pd.DataFrame, frame: int, verbose:bool, kalman_plot:Image):
        """
        Takes in autonomous car information and displays the images from the object and depth detection moddels,and
        other data about travel path

        Parameters
        ----------
        annotated_image : Image
        cropped_depth_images : Image
        imu_data :
        kalman_imu_data :
        frame : int
        verbose : bool

        Returns
        -------
        None
        """

        self.verbose = verbose

        # check if pause or play were clicked or if window closed
        """May need to relocate this"""
        event, values = self.window.read(timeout=1)

        # Reset objects no longer detected in frame
        #self.reset_depth_images(cropped_depth_images)

        # update main display_api with detected objects
        with io.BytesIO() as output:
            annotated_image.save(output, format="PNG")
            contents = output.getvalue()
        self.update_window("IMG", contents, annotated_image.size)

        with io.BytesIO() as output:
            cropped_depth_images.save(output, format="PNG")
            contents = output.getvalue()
        self.update_window("IMG1", contents, cropped_depth_images.size)

        # current picture frame
        self.update_window("frame", "Frame: " + str(frame))

        # current Speed and Kalman speed updated with api data
        self.grid_update(imu_data, kalman_imu_data)

        with io.BytesIO() as output:
            kalman_plot.save(output, format="PNG")
            contents = output.getvalue()
        self.update_window("IMG2", contents, kalman_plot.size)

        self.progress_bar.UpdateBar(frame + 1)
