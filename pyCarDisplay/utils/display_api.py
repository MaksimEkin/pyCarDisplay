import PySimpleGUI as sg
import os

class Display():
	def __init__(o, img_data_path, detected_images):
		o.close = sg.WIN_CLOSED
		o.IMG_PATH = img_data_path
		o.DET_IMG_PATH = detected_images
		o.speed = 20
		o.n_images = 0
		o.files = []

		#Just for demo
		o.wolf = o.DET_IMG_PATH+"wolf.png"

	def img(o, path, key):
		return sg.Image(path, key=key)

	def define_layout(o):
		return [
			[sg.Text("Frame: 1", size=(50, 1), key="frame")],
			[sg.Text("True Speed: "+str(o.speed), key="speed"), sg.Text("-- Kalman speed: "+str(o.speed), key="kspeed")],
			[o.img(o.wolf, "IMG")],
			[
				o.img(o.wolf, "IMG1"),
				o.img(o.wolf, "IMG2"),
				o.img(o.wolf, "IMG3"),
				o.img(o.wolf, "IMG4"),
				o.img(o.wolf, "IMG5")
			],
			[sg.ProgressBar(o.n_images, orientation='h', size=(50, 5), key='progressbar')],
			[sg.Button("Play"), sg.Button("Pause")]
		]

	def play(o):
		o.files = os.listdir(o.IMG_PATH)
		o.n_images = len(o.files)

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
		        window.FindElement("IMG").Update(o.IMG_PATH + "\\" + o.files[i % o.n_images])

				# This is just for demonstation purposes
		        if i == (o.n_images*3) // 4:
		            window.FindElement("IMG2").Update("")
		            window.FindElement("IMG3").Update("")
		            window.FindElement("IMG4").Update("")
		            window.FindElement("IMG5").Update("")

				# Also just for demonstation purposes
		        elif i == o.n_images // 4:
		            window.FindElement("IMG2").Update(o.DET_IMG_PATH + "wolf.png")
		            window.FindElement("IMG3").Update(o.DET_IMG_PATH + "leaf.png")


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
