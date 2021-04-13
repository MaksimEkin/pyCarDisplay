"""
This is the object detection module.

This code is borrowed from sgrvinod's GitHub repository named
a-PyTorch-Tutorial-to-Object-Detection, and modified to work with pyCarDisplay.
It preserved the main functionality of the unmodified version.

Reference:
    Vinodababu, S. (n.d.). A-PyTorch-Tutorial-to-Object-Detection. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

"""
from torchvision import transforms
from .__utils_obj_detection import *
from PIL import Image, ImageDraw, ImageFont
import sys
from termcolor import colored
import os


class ObjectDetection():
    def __init__(self,
                 model_path: str,
                 verbose: bool,
                 img_resize_size=(300, 300),
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225],
                 device="cpu"
                 ):
        """
        Initilize the object detection class.

        Parameters
        ----------
        model_path : str
            path to the pre-trained PyTorch model.
        verbose : bool
            Verbosity flag.
        img_resize_size : tuple, optional
            Image size during prediction. The default is (300, 300).
        norm_mean : list, optional
            model hyper-parameter. The default is [0.485, 0.456, 0.406].
        norm_std : list, optional
            model hyper-parameter. The default is [0.229, 0.224, 0.225].
        device : str, optional
            If 'gpu', model uses a CUDA device. The default is "cpu".

        Returns
        -------
        None.

        """
        sys.path.append(os.system("pwd"))
        self.verbose = verbose
        self.model_path = model_path

        # Use GPU if available
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                sys.exit("No cuda device found!")

        if self.verbose:
            print(colored("Object detection is using: " + str(self.device), "yellow"))

        # Load model checkpoint
        if self.verbose:
            print(colored("Object detection API is loading the model: " +
                          str(self.model_path), "blue"))

        checkpoint = torch.load(self.model_path, map_location=self.device)
        start_epoch = checkpoint['epoch'] + 1

        if self.verbose:
            print(colored('Loaded checkpoint from epoch %d.\n' %
                          start_epoch, "yellow"))

        self.model = checkpoint['model']
        self.model = self.model.to(device)
        self.model.eval()

        # Transforms
        self.resize = transforms.Resize(img_resize_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=norm_mean,
                                              std=norm_std)

    def detect(self,
               original_image: Image,
               suppress=None,
               min_score=0.2,
               max_overlap=0.5,
               top_k=200):
        """
        Predicts the objects in the image using the pre-trained model.

        Parameters
        ----------
        original_image : Image
            PIL image to be used for object detection.
        suppress : TYPE, optional
            DESCRIPTION. The default is None.
        min_score : float, optional
            min threshold for object to be classified. The default is 0.2.
        max_overlap : float, optional
            min threshold for object to be classified. The default is 0.5.
        top_k : int, optional
            how many objects to attempt to classify. The default is 200.

        Returns
        -------
        dict
            Returns a dictionary in format: {"annotated_image": annotated_image, "box_info": box_info, "detected": True}
            box_info is in format: box_info = {"text_size": text_sizes,"box_location": box_locations}, where the values
            are list of coordinates for the detection boxes.

        """

        # .show()
        return self.__detect_helper(original_image, min_score=min_score, max_overlap=max_overlap, top_k=top_k, suppress=suppress)

    def __detect_helper(self, original_image, min_score, max_overlap, top_k, suppress):
        """
        Performs the prediction.

        Parameters
        ----------
        original_image : PIL.Image
            Kitti dataset 1 frame image.
        min_score : float
            detection threshold.
        max_overlap : float
            detection threshold.
        top_k : int
            number of objects to search.
        suppress : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            Returns a dictionary in format: {"annotated_image": annotated_image, "box_info": box_info, "detected": True}
            box_info is in format: box_info = {"text_size": text_sizes,"box_location": box_locations}, where the values
            are list of coordinates for the detection boxes.

        """

        # Transform
        image = self.normalize(self.to_tensor(self.resize(original_image)))

        # Move to default device
        image = image.to(self.device)

        # Forward prop.
        predicted_locs, predicted_scores = self.model(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = self.model.detect_objects(predicted_locs,
                                                                      predicted_scores,
                                                                      min_score=min_score,
                                                                      max_overlap=max_overlap,
                                                                      top_k=top_k)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
            [original_image.width,
             original_image.height,
             original_image.width,
             original_image.height]
        ).unsqueeze(0)

        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l]
                      for l in det_labels[0].to('cpu').tolist()]

        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['background']:
            # Just return original image
            box_info = {
                "text_size": [],
                "box_location": []
            }
            return {"annotated_image": original_image, "box_info": box_info, "detected": False}

        # Annotate
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()

        # Suppress specific classes, if needed
        text_sizes = list()
        box_locations = list()
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location,
                           outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

            # Text
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2.,
                             box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]

            draw.rectangle(xy=textbox_location,
                           fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                      font=font)

            box_locations.append(box_location)
            text_sizes.append(text_size)

        del draw

        # the box information
        box_info = dict()
        box_info = {
            "text_size": text_sizes,
            "box_location": box_locations
        }

        return {"annotated_image": annotated_image, "box_info": box_info, "detected": True}
