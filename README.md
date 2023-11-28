# README

## Introduction
In this project, an overlay for a backup camera was created and includes lines to represent distances of 7ft, 10ft, and 15ft from the back of the vehicle. It also includes dynamic trajectory lines to assist the driver in determining the path of the vehicle at the maximum wheel angle.

## Prerequisites
1. **Python**: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
2. **OpenCV**: Install the OpenCV library by running the following command:
    ```
    pip install opencv-python
    ```
3. **NumPy**: Install NumPy using the following command:
    ```
    pip install numpy
    ```
## Usage
1. **Clone the Repository**: Clone this repository to your local machine using the following command:
    ```
    git clone https://github.com/joaquin-godinez/Backup-Camera-Overlay.git
    ```

2. **Navigate to the Code Directory**: Change your working directory to the folder containing the code:
    ```
    cd your-repository
    ```

3. **Replace the Video**: Replace the existing video file (`input_video.mp4`) in the code directory with your desired video. **Make sure to update the `video_path` variable in the code with the new video file path.**

4. **Run the Script**: Execute the Python script using the following command:
    ```
    python your_script_name.py
    ```

5. **View the Output**: The script will display the video with green lines, spline curves, and distance markers overlayed. Press the 'q' key to close the window.

6. **Check the Output Video**: The output video is saved as `output_video.avi` in the code directory.

## Notes
- Adjust the dimensions and parameters in the code according to your specific use case.
- Ensure that the input video is accessible and has the correct file path.
- The script uses the 'q' key to break the loop and close the video window.

