"""

"""
from torchvision import transforms
from .__utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy
import matplotlib
import easydict

class DepthDetection():
	 def __init__(self,
	 			data_dir:str,
                model_path:str,
				pretrained:bool,
				output_directory:str,
				input_height:int,
				input_width:int,
				model:str,
				mode:str,
				input_channels:int,
				num_workers:int):
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

    def detect(self,
               data_dir:str,
               model_path:str,
			   pretrained:bool,
			   output_directory:str,
			   input_height:int,
			   input_width:int,
			   model:str,
			   mode:str,
			   input_channels:int,
			   num_workers:int):
		pass
