"""
"""
from PIL import Image

class ImageProcessing():
    def __init__(self, car_images_path:str):
        self.img_path = car_images_path
        self.img = None

    def take_picture(self):
        return Image.open(self.img_path, mode='r').convert('RGB')

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

        for box in box_locations:
            x, y, h, w = box
            croped_depth_images.append(depth_image.copy().crop((x, y, x + w, y + h)))

        return cropped_depth_images
