"""

"""
from torchvision import transforms
from .__utils_depth_detection import *
from PIL import Image, ImageDraw, ImageFont
import numpy
import matplotlib
import easydict

"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

class DepthDetection():
	 def __init__(self, input_path:str, output_path:str, model_path:str, model_type="large", optimize=True):
        """

        Parameters
        ----------
        data_dir: path to the dataset folder
		model_path: path to save the trained model
		pretrained:
		output_directory: where save dispairities for tested images
		input_height
		input_width
		model: model for encoder (resnet18 or resnet50)
		mode: train or test
		input_channels Number of channels in input tensor (3 for RGB images)
		num_workers Number of workers to use in dataloader
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

    def run(input_path, output_path, model_path, model_type="large", optimize=True):
        """Run MonoDepthNN to compute depth maps.

        Args:
            input_path (str): path to input folder
            output_path (str): path to output folder
            model_path (str): path to saved model
        """
        print("initialize")

        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: %s" % device)

        # load network
        if model_type == "large":
            model = MidasNet(model_path, non_negative=True)
            net_w, net_h = 384, 384
        elif model_type == "small":
            model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
        else:
            print(f"model_type '{model_type}' not implemented, use: --model_type large")
            assert False

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
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

        model.eval()

        if optimize==True:
            rand_example = torch.rand(1, 3, net_h, net_w)
            model(rand_example)
            traced_script_module = torch.jit.trace(model, rand_example)
            model = traced_script_module

            if device == torch.device("cuda"):
                model = model.to(memory_format=torch.channels_last)  
                model = model.half()

        model.to(device)

        # get input
        img_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(img_names)

        # create output folder
        os.makedirs(output_path, exist_ok=True)

        print("start processing")

        for ind, img_name in enumerate(img_names):

            print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

            # input

            img = utils.read_image(img_name)
            img_input = transform({"image": img})["image"]

            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                if optimize==True and device == torch.device("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)  
                    sample = sample.half()
                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            # output
            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            return_val = utils.write_depth(filename, prediction, bits=2)

        print("finished")
        return return_val