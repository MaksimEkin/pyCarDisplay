import PySimpleGUI as sg
import pandas as pd
from PIL import Image
import io
from io import StringIO

class Display():

    # need to pass a frame dictionary that contains dictionaries of image paths and detected image lists
    def __init__(o, speed: int, total_frames: int):
        o.cropped_img_displayed = 5
        o.close = sg.WIN_CLOSED
        o.speed = speed
        o.total_frames = total_frames
        o.verbose = False
        # Create the window
        o.window = sg.Window("Autonomous Vehicle Object & Distance Detection", o.define_layout())
        o.progress_bar = o.window['progressbar']

    def img(o, path, key):
        return sg.Image(path, key=key)

    def format_pil_img(o, image: Image):
        return None
        '''
        image.show()

        with io.BytesIO() as output:
            tempBuff = StringIO()
            tempBuff.write(image)
            tempBuff.seek(0)
            Image.open(tempBuff)

            print("Line 26", image)
            image.save(output, format="PNG")
            contents = output.getvalue()
            return contents
        return io.BytesIO(image).getvalue()
        '''

    def update_window(o, key, data1, data2=''):
        if data2:
            o.window.FindElement(key).Update(data=data1, size=data2)
        else:
            o.window.FindElement(key).Update(data1)


    def depth_images_update(o, cropped_depth_images):
        for num, detected_image in enumerate(cropped_depth_images):
            if num < o.cropped_img_displayed:
                o.update_window("IMG" + str(num + 1), o.format_pil_img(detected_image), detected_image.size)


    def speed_update(o, imu_data, kalman_imu_data):
        if o.verbose:
            print("Examine imu=", imu_data['data'][0])
            print("Examine Kalman=", kalman_imu_data['data'][0])

        gold_speed = "True Speed: " + str(imu_data['data'][0])
        
        if o.verbose:
            print("gold_speed", gold_speed)
        o.window.FindElement("speed").Update(gold_speed)

        kalman_speed = "Kalman speed: " + str(kalman_imu_data['data'][0])
        o.update_window("kspeed", kalman_speed)


    def reset_depth_images(o, cropped_depth_images):
        for num in range(o.cropped_img_displayed):
            o.update_window("IMG" + str(num + 1), "")


    def define_layout(o):
        return [
            [sg.ProgressBar(o.total_frames, orientation='h', size=(50, 5), key='progressbar')],
            [sg.Text("Frame: 1", size=(50, 1), key="frame")],
            [sg.Text("True Speed:" + " " * 30 + str(o.speed), key="speed")],
            [sg.Text("Kalman speed:"+ " " * 20 + str(o.speed), key="kspeed")],
            [o.img("", "IMG")],
            [
                o.img("", "IMG1"),
                o.img("", "IMG2"),
                o.img("", "IMG3"),
                o.img("", "IMG4"),
                o.img("", "IMG5")
            ]
        ]

    def end(o):
        o.window.close()

    def play(o, annotated_image: Image, cropped_depth_images: list, imu_data: pd.DataFrame,
             kalman_imu_data: pd.DataFrame, frame: int, verbose:bool):
        
        
        cropped_depth_images = ['heh', 'ehh']
        
        o.verbose = verbose
        if o.verbose:
            print(type(annotated_image))

        # check if pause or play were clicked or if window closed
        """May need to relocate this"""
        event, values = o.window.read(timeout=1)

        # Reset objects no longer detected in frame
        o.reset_depth_images(cropped_depth_images)

        # update main display_api with detected objects
        #annotated_image.show()
        with io.BytesIO() as output:
            annotated_image.save(output, format="PNG")
            contents = output.getvalue()
        o.update_window("IMG", contents, annotated_image.size)

        # update up to cropped_img_displayed number of the depth images of detected objects
        ##for num, detected_image in enumerate(cropped_depth_images):
            ##if num < o.cropped_img_displayed:
                #o.update_window("IMG" + str(num + 1), detected_image, detected_image.size)

        # current picture frame
        o.update_window("frame", "Frame: " + str(frame))

        # current Speed and Kalman speed updated with api data
        o.speed_update(imu_data, kalman_imu_data)
        o.progress_bar.UpdateBar(frame + 1)
