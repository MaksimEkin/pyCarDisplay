# pyCarDisplay

<div align="center", style="font-size: 50px">
    <img src="https://github.com/MaksimEkin/pyCarDisplay/actions/workflows/unittests_ci.yml/badge.svg?branch=main"></img>
    <img src="https://img.shields.io/hexpm/l/plug"></img>
    <img src="https://img.shields.io/badge/python-v3.8.5-blue"></img>
</div>

<br>

Python Library for Simulating Autonomous Vehicle: pyCarDisplay.
pyCarDisplay is developed to read the Kitti dataset, and simulate an automated car.
It can perform object detection, depth detection, IMU sensor simulation, Kalman Filtering,
and display the results on a GUI.


## Installation
```shell
pip install pyCarDisplay
```
or

```shell
git clone https://github.com/MaksimEkin/pyCarDisplay
cd pyCarDisplay
python setup.py install
```

## Prerequisites
- Python >= v3.8.5
- Download the [pre-trained PyTorch object detection mode](https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe) (Provided by [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)).
- Download the [pre-trained PyTorch depth detection model](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt) (Provided by [OniroAI](https://github.com/OniroAI/MonoDepth-PyTorch))
- Download a set of raw sampels from the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php)


## Usage
```python
from pyCarDisplay.pyCarDisplay import CarDisplay

display = CarDisplay(
    car_images_path="../data/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/",
    imu_sensor_path='../data/2011_09_26/2011_09_26_drive_0005_sync/oxts/data/',
    object_detection_model_path='../data/checkpoint_ssd300.pth.tar',
    depth_detection_model_path='../data/model-f6b98070.pt',
    verbose=True,
    device="cpu"
)

display.start(verbose=True)
```

## Documentation
The documentation of **pyCarDisplay** can be found [here](https://maksimekin.github.io/pyCarDisplay/html/index.html)


## How to Cite pyCarDisplay?
```
@electronic{cmsc611_2021_umbc,
  author = {R. {Barron} and M. E. {Eren} and C. {Varga} and W. {Wang}},
  title = {pyCarDisplay},
  url = "https://github.com/MaksimEkin/pyCarDisplay"
}
```


## References
- Vinodababu, S. (n.d.). A-PyTorch-Tutorial-to-Object-Detection. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
- “MiDaS,” Pytorch.org. [Online]. Available: https://pytorch.org/hub/intelisl_midas_v2/. [Accessed: 27-Mar-2021].
- R. E. Kalman,  A New Approach to Linear Filtering and Prediction Problems, Research Institute for Advanced Study,2 Baltimore, Md.https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf
- A Geiger, P Lenz, C Stiller, and R Urtasun. 2013. Vision meets robotics: The KITTI dataset. Int. J. Rob. Res. 32, 11 (September 2013), 1231–1237. DOI:https://doi.org/10.1177/0278364913491297
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
