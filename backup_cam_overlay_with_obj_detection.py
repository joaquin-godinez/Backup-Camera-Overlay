import cv2
import numpy as np
import math

#for Yolov5
import torch
import torch.nn.functional as F
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import yaml

# Variables(10th Gen Honda Civic)
#referenced from "https://www.car.info/en-se/honda/civic/10th-generation-7898138/specs" and "https://www.gardnerweb.com/articles/developing-the-10th-generation-honda-civic"
carWidth = 2.076 #meters
overhangFront = 0.894 #meters
overhangRear = 1.03378 #meters
wheelBase = 2.7 #meters
trackFront = 1.547 #meters
trackRear = 1.576 #meters
turningCircle = 11.8 #meters diameter

rad = 0.017453292519943295
maxTurnAngle = math.atan2(wheelBase, (turningCircle - carWidth) / 2) / rad
#print(maxTurnAngle) # max turn angle is 29 degrees

#maximum turning rate
turningRate = math.tan(maxTurnAngle * rad) / wheelBase / rad # degrees per meter
#print(turningRate)# turning rate = 11.78 deg/m

#turning radius at maximum angle
TR = wheelBase/math.sin(maxTurnAngle*math.pi/180)
#print(TR) #turning radius = 5.56 meters --> represents the turning radius of outer front tire

def Obj_detection(frame, model, input_size):

    # Transformation for normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Resize frame
    frame = cv2.resize(frame, input_size)

    # Convert BGR to RGB
    frame = frame[:, :, ::-1]

    # Convert to PyTorch tensor
    img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    # Normalize
    img = normalize(img)
    # Perform inference using YOLOv5 model
    with torch.no_grad():
        pred = model(img)[0]

    print("Inference completed!")

    # Post-processing
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.4)[0]
    obj_frame = frame.astype(np.uint8)
    return pred, obj_frame


def backup_overlay_with_obj_detect(video_path):
    #load class names
    coco_yaml_path = r"C:\Users\joaqu\OneDrive\Desktop\Python\yolov5\data\coco.yaml"

    # Load the YAML file with explicit encoding
    with open(coco_yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # Extract the class names
    class_names = data['names']

    # Load YOLOv5
    weights_path = r"C:\Users\joaqu\OneDrive\Desktop\Python\yolov5s.pt"
    device = torch.device("cpu")
    model = attempt_load(weights_path, device)
    print("Model loaded successfully!")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # YOLOv5 input size
    input_size = (640, 480)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # print(frame_height)
    # print(frame_width)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))


    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break
        #------------------spline formation------------------------------------------------------------------------------------------------------------------------------
        # Draw a curved line (quadratic Bézier curve)
        curve_color = (255, 200, 255)  # BGR format
        curve_thickness = 3

        # Define control points for the quadratic Bézier curve
        p0_left = (0, frame_height * 91/100)
        p1_left = (405, 303)
        p2_left = (frame_width, 303)

        p0_right = (frame_width, frame_height * 91/100)
        p1_right = (1280-405, 303)
        p2_right = (0, 303)
            
        # Generate points along the left curve   
        t = np.linspace(0, 1, 100)
        curve_points_left = np.array([(1 - ti) ** 2 * np.array(p0_left) + 2 * (1 - ti) * ti * np.array(p1_left) + ti ** 2 * np.array(p2_left) for ti in t], np.int32)
            
        # Generate points along the right curve       
        curve_points_right= np.array([(1 - ti) ** 2 * np.array(p0_right) + 2 * (1 - ti) * ti * np.array(p1_right) + ti ** 2 * np.array(p2_right) for ti in t], np.int32)
    
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Draw two vertical lines in the center
        line_thickness = 5
        line1_start = (frame_width * 28 // 100, frame_height *5 // 10)
        line1_end = (0, frame_height * 91 // 100)
        line2_start = (frame_width-line1_start[0],frame_height * 5// 10)
        line2_end = (frame_width, frame_height * 91 // 100)

        # Draw a horizontal line connecting the top tips of the vertical lines representing 15 feet FROM REAR WHEEL
        rect_line_start = (line1_start[0], line1_start[1])
        rect_line_end = (line2_start[0], line2_start[1])
            
        line_vert_difference = (line1_end[1]-line1_start[1])
        #print(line1_start[1] + ((54*line_vert_difference) // 100))

        line_horiz_difference = (line1_end[0]-line1_start[0])
        # print(line_horiz_difference*54/100 + line1_start[0])
        
        # Draw a horizontal line for 10 feet FROM REAR WHEEL
        mid_rect_line1_start = (line2_start[0] - ((54*line_horiz_difference) // 100), line1_start[1] + ((54*line_vert_difference) // 100)) #(width, height)
        mid_rect_line1_end = (line1_start[0] + ((54*line_horiz_difference) // 100), line1_start[1] + ((54*line_vert_difference) // 100)) #(width, height)

        # Draw a horizontal line for 7 feet FROM REAR WHEEL
        mid_rect_line2_start = (line2_start[0] - ((99*line_horiz_difference) // 100), line1_start[1] + ((99*line_vert_difference) // 100))
        mid_rect_line2_end = (line1_start[0] + ((99*line_horiz_difference) // 100) , line1_start[1] + ((99*line_vert_difference) // 100))

        #object detection
        pred, obj_frame= Obj_detection(frame, model, input_size)


        if pred is not None:
            # Draw bounding boxes
            for det in pred:
                xmin, ymin, xmax, ymax, conf, cls = det.cpu().numpy()
                color = (0, 255, 0)  # Green color for bounding boxes
                cv2.rectangle(frame, (int(xmin*2), int(ymin*1.5)), (int(xmax*2), int(ymax*1.5)), color, 2) # the coefficients in the xmax and ymax account for the frame size difference converts 640x480 to 1280x720
                
                cv2.putText(frame, f"{class_names[int(cls)]} {conf:.2f}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the lines on the frame
        color = (0, 255, 0)  # Green color in BGR format
        color1 = (0, 255, 255)
        color2 = (0, 0 ,255)
        
        cv2.line(frame, line1_start, line1_end, color, line_thickness)
        cv2.line(frame, line2_start, line2_end, color, line_thickness)
        cv2.line(frame, rect_line_start, rect_line_end, color, line_thickness)
        cv2.line(frame, mid_rect_line1_start, mid_rect_line1_end, color1, line_thickness)
        cv2.line(frame, mid_rect_line2_start, mid_rect_line2_end, color2, line_thickness)

        # Draw the curved line on the frame
        cv2.polylines(frame, [curve_points_left], isClosed=False, color=curve_color, thickness=curve_thickness) #spline 
        cv2.polylines(frame, [curve_points_right], isClosed=False, color=curve_color, thickness=curve_thickness) #spline

        # Combine the original frame with the green overlay
        result_frame = cv2.addWeighted(frame, 1, frame, 0.5, 0)

        # Display the result
        cv2.imshow('Backup_cam', result_frame)
        
        # Write the frame to the output video
        out.write(result_frame)

        
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # video_path = r"C:\Users\joaqu\OneDrive\Pictures\Camera Roll\WIN_20231201_16_43_49_Pro.mp4"  # Change this to your input video file
    backup_overlay_with_obj_detect(video_path)
    






    

