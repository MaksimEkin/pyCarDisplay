"""
Creates Kalman Filter plot.
"""
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image

class KalmanPlot():
    
    def __init__(self, img_resize_size=(300, 300), pixel_sizes=[1242, 375], dpi=100):
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
        
        plt.figure(figsize=(self.pixel_sizes[0]/self.dpi), dpi=self.dpi)
        plot = plt.plot(self.kalman_predictions, label='Kalman signal')
        plot = plt.plot(self.true_values, label='Ground truth')
        
        plt.legend(fontsizeint="large")
        plt.tight_layout()
        
        plt.ion()
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return np.array(Image.open(buf))