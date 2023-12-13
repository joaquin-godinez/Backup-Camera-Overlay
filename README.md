# README

## Backup Camera Overlay

### Introduction
In this project, an overlay for a backup camera was created, including lines to represent distances of 7ft, 10ft, and 15ft from the back of the vehicle. It also includes dynamic trajectory lines to assist the driver in determining the path of the vehicle at the maximum wheel angle. If object detection is desired, please follow instructions under the "backup camera overlay with object detection" section **after installing the necessary libraries in the pre-requisites section below**. 

### Pre-requisites
1. **Python**: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
2. **OpenCV**: Install the OpenCV library by running the following command:
    ```
    pip install opencv-python
    ```
3. **NumPy**: Install NumPy using the following command:
    ```
    pip install numpy
    ```

### Usage
1. **Clone the Repository**: Clone this repository to your local machine using the following command:
    ```
    git clone https://github.com/joaquin-godinez/Backup-Camera-Overlay.git
    ```

2. **Navigate to the Code Directory**: Change your working directory to the folder containing the code:
    ```
    cd Backup-Camera-Overlay
    ```

3. **Replace the Video**: Replace the existing video file (`input_video.mp4`) in the code directory with your desired video. **Make sure to update the `video_path` variable in the code with the new video file path.**

4. **Run the Script**: Execute the Python script using the following command:
    ```
    python backup_camera_overlay.py
    ```

5. **View the Output**: The script will display the video with green lines, spline curves, and distance markers overlayed. Press the 'q' key to close the window.

6. **Check the Output Video**: The output video is saved as `output_video.avi` in the code directory.

7. **Notes**
    - Adjust the dimensions and parameters in the code according to your specific use case.
    - Ensure that the input video is accessible and has the correct file path.
    - The script uses the 'q' key to break the loop and close the video window.

## Backup Camera Overlay with Object Detection

### Introduction
In this extended version of the project, object detection capabilities have been added to the backup camera overlay using PyTorch and YOLOv5. This allows for the identification of objects in the camera's field of view, enhancing safety during vehicle reversing.

### Additional Prerequisites
4. **PyTorch**: Install PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).
   
5. **PyYAML**: Install PyYAML using the following command:
    ```
    pip install pyyaml
    ```

6. **YOLOv5**: Clone the YOLOv5 repository to your local machine using the following command:
    ```
    git clone https://github.com/ultralytics/yolov5.git
    ```
    - Navigate to the YOLOv5 directory:
        ```
        cd yolov5
        ```
    - Install YOLOv5 dependencies:
        ```
        pip install -U -r requirements.txt
        ```
7. **Download COCO Weights**: To use pre-trained COCO weights with YOLOv5, download them using the following command:
    ```
    python -c "from pathlib import Path; Path('weights').mkdir(parents=True, exist_ok=True)"
    python -m wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt -O weights/yolov5s.pt
    ```

8. **Install COCO Data**: Install the COCO data requirements using the following command:
    ```
    python -m pip install -U 'git+https://github.com/ultralytics/yolov5.git#egg=yolov5[utils]'
    ```


### Additional Usage Instructions
1. Follow steps 1-3 from the "Backup Camera Overlay" section.
2. Run the extended script using the following command:
    ```
    python backup_camera_overlay_with_object_detection.py
    ```

3. The script will display the video with object detection bounding boxes, distance markers, and trajectory lines. Press the 'q' key to close the window.

4. The output video is saved as `output_video_with_detection.avi` in the code directory.

5. Adjust parameters and model configurations according to your specific requirements.

### Notes
- Fine-tune object detection parameters based on your use case.
