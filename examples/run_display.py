from pyCarDisplay.utils.display_api import Display
import os

DET_IMG_PATH = ""
IMG_PATH = ""

wolf = DET_IMG_PATH+"wolf.png"
leaf = DET_IMG_PATH+"leaf.png"
img_files = os.listdir(IMG_PATH)

image_frames_dict = {}

i = 0
for image in img_files:
    if i < 20:
        image_frames_dict[str(i)] = {"image":IMG_PATH+image, 'detected_object_images':[ wolf, leaf]}
    elif i < 40:
        image_frames_dict[str(i)] = {"image":IMG_PATH+image, 'detected_object_images': [wolf, wolf, leaf]}
    elif i < 60:
            image_frames_dict[str(i)] = {"image":IMG_PATH+image, 'detected_object_images': []}
    else:
        image_frames_dict[str(i)] = {"image":IMG_PATH+image, 'detected_object_images': [leaf]}
    i+=1

car_disp = Display(image_frames_dict)
car_disp.play()
