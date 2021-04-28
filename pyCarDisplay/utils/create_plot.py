"""
Creates Kalman Filter plot.
"""
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image


class KalmanPlot():

    def __init__(self, img_resize_size=(300, 300), pixel_sizes=[1242, 375], dpi=100, track_n_frames=10, plot_column="af"):
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
        track_n_frames : int, optional
            How many frames to track in the Kalman plot. The default is 10.
        plot_column : str, optional
            Which IMU column to plot in the Kalman plot. The default is "af".
        """

        self.kalman_predictions = list()
        self.true_values = list()
        self.img_resize_size = img_resize_size
        self.pixel_sizes = pixel_sizes
        self.dpi = dpi
        self.track_n_frames = track_n_frames
        self.plot_column = plot_column

    def gen_plot(self, prediction, true_value):
        """
        Create a plot of kalman prediction and true value over time.

        Parameters
        ----------
        prediction : float
            Kalman prediction.
        true_value : float
            True sensor value.
        """
        self.kalman_predictions.append(prediction)
        self.true_values.append(true_value)

        if len(self.kalman_predictions) > self.track_n_frames:
            self.kalman_predictions.pop(0)
            self.true_values.pop(0)

        plt.ioff()

        plt.figure(figsize=(
            self.pixel_sizes[0]/self.dpi, self.pixel_sizes[1]/self.dpi), dpi=self.dpi)

        plot = plt.plot(self.kalman_predictions, label='Kalman signal',
                        linestyle="solid", color="#cc5f43")
        plot = plt.plot(self.true_values, label='Ground truth',
                        linestyle="dashed", color="#8176cc")

        plt.title("Kalman Plot for: " + str(self.plot_column),
                  fontsize="large", color="white")
        plt.legend(fontsize="large")
        plt.tight_layout()

        plt.ion()

        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)

        return Image.fromarray(np.array(Image.open(buf)))
