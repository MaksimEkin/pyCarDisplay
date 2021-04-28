"""
Creates Kalman Filter plot.
"""
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image

class KalmanPlot():

    def __init__(self, img_resize_size=(300, 300), pixel_sizes=[1242, 375], dpi=100):
        """
        Initilize the KalmanPlot class.

        Parameters
        ----------

        img_resize_size : tuple, optional
            Size of the image when performing object detection. The default is (300, 300).
        pixel_sizes : list, optional
            The sizes of the pixels. The default is [1242, 375].
        dpi : int, optional
            Dots per inch. The default is 100.
        """

        self.kalman_predictions = list()
        self.true_values = list()
        self.img_resize_size = img_resize_size
        self.pixel_sizes = pixel_sizes
        self.dpi = dpi


    def gen_plot(self, prediction, true_value):
        """

        """
        self.kalman_predictions.append(prediction)
        self.true_values.append(true_value)

        plt.ioff()

        plt.figure(figsize=(self.pixel_sizes[0]/self.dpi, self.pixel_sizes[1]/self.dpi), dpi=self.dpi)
        plot = plt.plot(self.kalman_predictions, label='Kalman signal')
        plot = plt.plot(self.true_values, label='Ground truth')

        plt.title("Kalman Plot for Forward Acceleration", fontsize="large", color="white")
        plt.legend(fontsize="large")
        plt.tight_layout()

        plt.ion()

        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)

        return Image.fromarray(np.array(Image.open(buf)))
