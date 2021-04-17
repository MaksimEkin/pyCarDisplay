"""
This is the depth detection module.

We borrowed this code from the MiDaS GitHub repository, authored by intel-isl.
We then modified the code to work with the pyCarDisplay library. This code
obtains an image and turns it into a colorized heat map based on the depth
of objects within the environment. Original functionality is preserved, but
functionality is wrapped within a class with new additions.

Reference:
    “MiDaS,” Pytorch.org. [Online]. Available: https://pytorch.org/hub/intelisl_midas_v2/. [Accessed: 27-Mar-2021].

Original license from MiDaS code is below.

MIT License

Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
from .midas.midas_net import MidasNet
from .midas.midas_net_custom import MidasNet_small
from .midas.transforms import Resize, NormalizeImage, PrepareForNet
import sys
from termcolor import colored
import matplotlib.pyplot as plt
import io
from PIL import Image



class DepthDetection():
    def __init__(self, verbose: bool, model_path: str, model_type="large",
                 optimize=True, model=None, device="cpu", transform=None, dpi=100,
                 alpha=0.6, pixel_sizes=[1242, 375]):
        """

        Initialize the depth detection class.

        Parameters
        ----------
        verbose : bool
            Print out information regarding API activity.
        model_path : str
            The path to the pretrained model file.
        model_type : str, optional
            The type of model to use to detect depth. The default is "large".
        optimize : bool, optional
            Optimize the depth detection. The default is True.
        model : None, optional
            If passed in, this is the predefined model. The default is None.
        device : str, optional
            Use CUDA device if available. The default is "cpu".
        transform : None, optional
            Placeholder for the transform class variable. The default is None.
        dpi : int, optional
            Dots per inch. The default is 100.
        alpha : double, optional
            The alpha variable. The default is 0.6.
        pixel_sizes : list, optional
            The sizes of the pixels. The default is [1242, 375].

        Returns
        -------
        None.

        """

        self.model_path = model_path
        self.model_type = model_type
        self.optimize = optimize
        self.verbose = verbose
        self.dpi = dpi
        self.alpha = alpha
        self.pixel_sizes = pixel_sizes

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
            print(colored("Depth detection is using: " + str(self.device), "yellow"))

        # load network
        if model_type == "large":
            self.model = MidasNet(model_path, non_negative=True)
            self.net_w, self.net_h = 384, 384
        elif model_type == "small":
            self.model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3",
                                        exportable=True, non_negative=True, blocks={'expand': True})
            self.net_w, self.net_h = 256, 256  # Self-ize these variables
        else:
            if verbose:
                print(
                    f"model_type '{model_type}' not implemented, use: --model_type large")
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
                NormalizeImage(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def run(self, verbose: bool, pil_image: Image, optimize=True):
        """


        Parameters
        ----------
        verbose : bool
            Print out information regarding API activitty.
        pil_image : Image
            The image to be evaluated.
        optimize : bool, optional
            Optimize the depth detection. The default is True.

        Returns
        -------
        prediction : Image
            An enhanced image with the colorized depth predictions of the objects within the environment.

        """

        self.model.to(self.device)

        num_images = 1

        pil_arr_image = np.array(pil_image)

        img_input = self.transform({"image": pil_arr_image})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=pil_arr_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return Image.fromarray(self.convert_to_heat_map(prediction, pil_image))

    def convert_to_heat_map(self, prediction, original_image):
        """


        Parameters
        ----------
        prediction : Image
            The depth predicitons of the objects withi the enviroment.
        original_image : Image
            The original image.

        Returns
        -------
        np.array()
            A numpy array of the color values of each of the pixels in the image.

        """

        plt.ioff()

        fig, ax = plt.subplots(figsize=(self.pixel_sizes[0]/self.dpi, self.pixel_sizes[1]/self.dpi), dpi=self.dpi)
        figure = ax.imshow(original_image)
        figure = ax.imshow(prediction, alpha=self.alpha, cmap='nipy_spectral')

        plt.tight_layout()
        plt.ion()

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)

        return np.array(Image.open(buf))
