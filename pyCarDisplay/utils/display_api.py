import PySimpleGUI as sg
#import os

class Display():

	# need to pass list of
	def __init__(o, image_frames_dict):
		o.close = sg.WIN_CLOSED

		#o.IMG_PATH = img_data_path
		#o.DET_IMG_PATH = detected_images
		o.speed = 0
		#o.files = []

		o.image_frames_dict = image_frames_dict
		o.n_images = len(image_frames_dict)

	def img(o, path, key):
		return sg.Image(path, key=key)

	def define_layout(o):
		return [
            [sg.Button("Play"), sg.Button("Pause")],
            [sg.ProgressBar(o.n_images, orientation='h', size=(50, 5), key='progressbar')],
			[sg.Text("Frame: 1", size=(50, 1), key="frame")],
			[sg.Text("True Speed: " + str(o.speed), key="speed"), sg.Text("-- Kalman speed: " + str(o.speed), key="kspeed")],
			[o.img("", "IMG")],
			[
				o.img("", "IMG1"),
				o.img("", "IMG2"),
				o.img("", "IMG3"),
				o.img("", "IMG4"),#
				o.img("", "IMG5")
			]
		]

	def play(o):
		#o.files = os.listdir(o.IMG_PATH)
		image_frames_dict = {
		'0': {'image':'path/to/image', 'detected_object_images': ['path/to/detected/image1', 'path/to/detected/image1'] },
		'1': {'image':'path/to/image', 'detected_object_images': ['path/to/detected/image1'] }
		}

		# Create the window
		window = sg.Window("Autonomous Vehicle Object & Distance Detection", o.define_layout())
		progress_bar = window['progressbar']

		# Create an event loop
		i = 0
		while i < o.n_images:

			# check if pause or play were clicked or if window closed
			event, values = window.read(timeout=10)

			# Window closed
			if event == o.close:
				i = o.n_images

			if event == 'Pause':
				while event != "Play" and event != o.close:

					# Read the window until the user clicks play or closes the window
					event, values = window.read(timeout=100)
					if event == o.close:
						i = o.n_images

			# Default is to play the images
			else:

				# update main display
				print("IMG = ",o.image_frames_dict[str(i)]["image"])
				window.FindElement("IMG").Update(o.image_frames_dict[str(i)]["image"])



				# update up to 5 of the images of detected objects
				num = 1
				for detected_image in o.image_frames_dict[str(i)]["detected_object_images"]:
					if num < 5:
						window.FindElement("IMG"+str(num)).Update(detected_image)
						num += 1

				# Reset objects no longer detected in frame
				while num <=5:
					window.FindElement("IMG"+str(num)).Update("")
					num +=1


				window.FindElement("frame").Update("Frame: "+str(i + 1))

				# Speed and Kalman speed need to be updated with api data
				window.FindElement("speed").Update("True Speed: "+str(i + 3))
				window.FindElement("kspeed").Update("-- Kalman speed: " + str(i + 9))

				progress_bar.UpdateBar(i + 1)
				i += 1

				# Loops the image play
				if i + 1 == o.n_images:
					i = 0

		window.close()
