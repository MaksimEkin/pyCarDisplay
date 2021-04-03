import PySimpleGUI as sg
from PIL import Image
import pandas as pd

class Display():

	# need to pass a frame dictionary that contains dictionaries of image paths and detected image lists
	def __init__(o, speed:int, total_frames:int):
		o.cropped_img_displayed = 5
		o.close = sg.WIN_CLOSED
		o.speed = speed
		o.total_frames = total_frames

		# Create the window
		o.window = sg.Window("Autonomous Vehicle Object & Distance Detection", o.define_layout())
		o.progress_bar = o.window['progressbar']

	def img(o, path, key):
		return sg.Image(path, key=key)

	def define_layout(o):
		return [
			[sg.ProgressBar(o.total_frames, orientation='h', size=(50, 5), key='progressbar')],
			[sg.Text("Frame: 1", size=(50, 1), key="frame")],
			[sg.Text("True Speed: " + str(o.speed), key="speed"), sg.Text("-- Kalman speed: " + str(o.speed), key="kspeed")],
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

	def play(o, annotated_image:Image, cropped_depth_images:list, depths:list, imu_data:pd.DataFrame, kalman_imu_data:pd.DataFrame, frame:int):
		cropped_depth_images = ['heh', 'ehh']

		# Reset objects no longer detected in frame
		for num in range(o.cropped_img_displayed):
			o.window.FindElement("IMG" + str(num + 1)).Update("")

		# check if pause or play were clicked or if window closed
		event, values = o.window.read(timeout = 10)

		# update main display_api
		o.window.FindElement("IMG").Update(annotated_image)

		# update up to cropped_img_displayed number of the depth images of detected objects
		for num, detected_image in enumerate(cropped_depth_images):
			if num < o.cropped_img_displayed:
				o.window.FindElement("IMG" + str(num + 1)).Update(detected_image)

		o.window.FindElement("frame").Update("Frame: " + str(frame))

		# Speed and Kalman speed need to be updated with api data
		o.window.FindElement("speed").Update("True Speed: " + str(imu_data.speed.values()[0]))
		o.window.FindElement("kspeed").Update("-- Kalman speed: " + str(kalman_imu_data.speed.values()[0]))

		o.progress_bar.UpdateBar(frame + 1)
