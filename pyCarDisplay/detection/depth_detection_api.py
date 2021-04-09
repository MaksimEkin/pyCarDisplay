"""
Compute depth maps for images in the input folder.
"""
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib
import cv2
import os
import glob
import torch
import argparse
from torchvision.transforms import Compose
from .midas.midas_net import MidasNet
from .midas.midas_net_custom import MidasNet_small
from .midas.transforms import Resize, NormalizeImage, PrepareForNet
import sys

class DepthDetection():
    def __init__(self, verbose:bool, model_path:str, model_type="large", optimize=True, model=None, device="cpu", transform=None):
        """

        Parameters
        ----------
        verbose:Prints out information regarding API activity is set to True.
        model_path:The path to the depth detection ML model file.
        model_type:Set to either large or small. Option must match the file specified in model_path.
        optimize:Currently does nothing.
        model:If specified, defines the model to use for depth detection.
        
        Returns
        -------
        None.
        """

        self.model_path = model_path
        self.model_type = model_type
        self.optimize = optimize
        # select device
        # Use GPU if available
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                sys.exit("No cuda device found!")
        
        if self.verbose:
            print("Object detection is using: " + str(self.device))

        # load network
        if model_type == "large":
            self.model = MidasNet(model_path, non_negative=True)
            self.net_w, self.net_h = 384, 384
        elif model_type == "small":
            self.model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            self.net_w, self.net_h = 256, 256 # Self-ize these variables
        else:
            if verbose:
                print(f"model_type '{model_type}' not implemented, use: --model_type large")
            assert False

        self.transform = Compose(
            [
                Resize(
                    self.net_w,
                    self.net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def run(self, verbose:bool, pil_image:Image, optimize=True):
        """Run MiDaS to compute depth maps.

        Args:
            input_path (str): path to input folder
            model_path (str): path to saved model
        """
        if verbose:
            print("initialize")

        self.model.to(self.device)

        num_images = 1

        if verbose:
            print("start processing")

        pil_image=np.array(pil_image)

        img_input = self.transform({"image": pil_image})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if optimize==True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=pil_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        if verbose:
            print("finished")
        return prediction
