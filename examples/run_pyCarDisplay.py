# IMPORT
from pyCarDisplay.pyCarDisplay import CarDisplay

display = CarDisplay(
    # Kitti dataset:
    # https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip
    car_images_path="../data/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/", 
    imu_sensor_path='../data/2011_09_26/2011_09_26_drive_0005_sync/oxts/data/',
    # Object detection model downloaded from:
    # https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe
    object_detection_model_path='../data/checkpoint_ssd300.pth.tar',
    # Depth detection model downloaded from:
    # https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt
    depth_detection_model_path='../data/model-f6b98070.pt',
    # Print status
    verbose=True,
    # Run on CPU
    device="cpu"
)

# START
display.start(verbose=True)