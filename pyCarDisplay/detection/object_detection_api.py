"""

"""
from torchvision import transforms
from .__utils_obj_detection import *
from PIL import Image, ImageDraw, ImageFont

class ObjectDetection():
    def __init__(self,
                 model_path:str,
                 verbose:bool,
                 img_resize_size=(300, 300),
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225],
                 ):
        """


        Parameters
        ----------
        model_path : str
            DESCRIPTION.
        verbose : bool
            DESCRIPTION.
        img_resize_size : TYPE, optional
            DESCRIPTION. The default is (300, 300).
        norm_mean : TYPE, optional
            DESCRIPTION. The default is [0.485, 0.456, 0.406].
        norm_std : TYPE, optional
            DESCRIPTION. The default is [0.229, 0.224, 0.225].
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1

        if self.verbose:
            print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)

        self.model = checkpoint['model']
        self.model = self.model.to(device)
        self.model.eval()


        # Transforms
        self.resize = transforms.Resize(img_resize_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=norm_mean,
                                              std=norm_std)

    def detect(self,
               original_image:Image,
               suppress:list,
               min_score=0.2,
               max_overlap=0.5,
               top_k=200):
        """


        Parameters
        ----------
        original_image : str
            DESCRIPTION.
        suppress : list
            DESCRIPTION.
        min_score : TYPE, optional
            DESCRIPTION. The default is 0.2.
        max_overlap : TYPE, optional
            DESCRIPTION. The default is 0.5.
        top_k : TYPE, optional
            DESCRIPTION. The default is 200.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self.__detect(original_image, min_score=min_score, max_overlap=max_overlap, top_k=top_k) # .show()


    def __detect_helper(self, original_image, min_score, max_overlap, top_k, suppress=None):
        """
        Detect objects in an image with a trained SSD300, and visualize the results.

        :param original_image: image, a PIL Image
        :param min_score: minimum threshold for a detected box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
        :return: {"annotated_image":annotated_image, "box_info":box_info}, annotated_image is a PIL Image
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
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['background']:
            # Just return original image
            return original_image

        # Annotate
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./calibril.ttf", 15)

        # Suppress specific classes, if needed
        text_sizes = list()
        box_locations = list()
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

            # Text
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]

            if self.verbose:
                print(str(text_size[0]) + ' * ' + str(text_size[1]) + '\nx = ' + str(box_location[0]) + '\ny = ' + str(box_location[1]))

            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                      font=font)

            box_locations.append(box_location)
            text_sizes.append(text_size)

        del draw

        # the box information
        box_info = dict()
        box_info = {
            "text_size":text_sizes,
            "box_location":box_locations
            }

        return {"annotated_image":annotated_image, "box_info":box_info}
