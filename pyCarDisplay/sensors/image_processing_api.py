"""
"""
import glob
from PIL import Image

class ImageProcessing():
    def __init__(self, car_images_path:str, image_extension="png"):
        self.images_directory = car_images_path
        self.all_images_path = glob.glob(str(self.images_directory)+"*."+str(image_extension))
        #ini_list.sort(key = lambda x: x.split()[1])

    def take_picture(self, frame:int):
        return Image.open(self.all_images_path[frame], mode='r').convert('RGB')

    def synch_objDet_depDet_data(box_locations, depth_image):
        """Takes in object locations in a snapshot and the depth image to return depth heatmap of the objects.

        Args:
            param1: list of the object locations found in object detection machine learning model.
            param2: image returned from depth detection machine learning model.

        Returns:
            list of cropped images at the location of the object detection
            and the depth rendering of the depth detection model.
        """

        croped_depth_images = []
    
        if box_locations:
            for box in box_locations:
                x, y, h, w = box
                croped_depth_images.append(depth_image.copy().crop((x, y, x + w, y + h)))

        return cropped_depth_images
