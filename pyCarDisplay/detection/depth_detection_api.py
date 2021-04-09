"""

"""
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy
import matplotlib

"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import argparse
from torchvision.transforms import Compose
from .midas.midas_net import MidasNet
from .midas.midas_net_custom import MidasNet_small
from .midas.transforms import Resize, NormalizeImage, PrepareForNet

class DepthDetection():
    def __init__(self, verbose:bool, model_path:str, model_type="large", optimize=True):
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

        self.model_path = model_path
        self.model_type = model_type
        self.optimize = optimize
        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print("device: %s" % device)

        # load network
        if model_type == "large":
            model = MidasNet(model_path, non_negative=True)
            net_w, net_h = 384, 384
        elif model_type == "small":
            model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256 # Self-ize these variables
        else:
            if verbose:
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

        #model.eval()

    def run(self, verbose:bool, pil_image:Image, optimize=True):
        """Run MiDaS to compute depth maps.

        Args:
            input_path (str): path to input folder
            model_path (str): path to saved model
        """
        if verbose:
            print("initialize")



        if optimize==True:
            rand_example = torch.rand(1, 3, net_h, net_w)
            model(rand_example)
            traced_script_module = torch.jit.trace(self.model, rand_example)
            self.model = traced_script_module

            if device == torch.device("cuda"):
                self.model = self.model.to(memory_format=torch.channels_last)  
                self.model = self.model.half()

        self.model.to(device)

        # get input
        #img_names = glob.glob(os.path.join(input_path, "*"))
        num_images = 1 #len(img_names)

        # create output folder
        #os.makedirs(output_path, exist_ok=True)

        if verbose:
            print("start processing")

        #for ind, img_name in enumerate(img_names):

            #print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

            # input

            #img = utils.read_image(img_name)
        # use numpy to convert the pil_image into a numpy array
        pil_image=np.array(pil_image)

        # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
        # the color is converted from RGB to BGR format
        #img=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_input = transform({"image": pil_image})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize==True and device == torch.device("cuda"):
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

        # output
        #filename = os.path.join(
        #    output_path, os.path.splitext(os.path.basename(img_name))[0]
        #)
        #return_val = utils.write_depth(img_input, prediction, bits=2)

        #color_coverted = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        #pil_image=Image.fromarray(color_coverted)
        if verbose:
            print("finished")
        return prediction
