.. pyCarDisplay documentation master file, created by
   sphinx-quickstart on Thu Apr  8 23:23:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyCarDisplay's documentation!
========================================

Python Library for Simulating Autonomous Vehicle: pyCarDisplay.
pyCarDisplay is developed to read the Kitti dataset, and simulate an automated car.
It can perform object detection, depth detection, IMU sensor simulation, Kalman Filtering,
and display the results on a GUI.


Prerequisites
========================================
- Python >= v3.8.5
- Trained PyTorch Object Detection Model: https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe (Provided by [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection))
- Trained PyTorch Depth Detection Model: https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt (Provided by [OniroAI](https://github.com/OniroAI/MonoDepth-PyTorch))
- Kitti Dataset Samples: http://www.cvlibs.net/datasets/kitti/raw_data.php


Installation
========================================
.. code-block:: console

   git clone https://github.com/MaksimEkin/pyCarDisplay
   cd pyCarDisplay
   pip install pyCarDisplay

or

.. code-block:: console

    git clone https://github.com/MaksimEkin/pyCarDisplay
    cd pyCarDisplay
    python setup.py install

The simulator also needs, at minimum, the pre-trained models and Kitti dataset
samples. Please see the prerequisites.


Example Usage
========================================
.. code-block:: python

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


References
========================================
1- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
2- https://github.com/OniroAI/MonoDepth-PyTorch


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
