#!/usr/bin/env python3
"""
ROS node for lane detection on CARLA dashcam using Ultra-Fast-Lane-Detection model
Subscribes to CARLA camera topic and publishes lane detection results
"""
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image as PILImage
import scipy.special

# Import model from the cloned repository
from model.model import parsingNet

class CarlaLaneDetectionNode:
    def __init__(self):
        rospy.init_node('carla_lane_detection', anonymous=True)
        
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        # CULane configuration (matching the pretrained model)
        self.griding_num = 200
        self.cls_num_per_lane = 18
        self.culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
        self.culane_img_w, self.culane_img_h = 1640, 590
        
        # Load pretrained model
        model_path = rospy.get_param('~model_path', 'modelPath/culane_18.pth')
        rospy.loginfo(f"Loading model from: {model_path}")
        
        self.net = parsingNet(pretrained=False, backbone='18', 
                             cls_dim=(self.griding_num+1, self.cls_num_per_lane, 4), 
                             use_aux=False)
        
        state_dict = torch.load(model_path, map_location=self.device)['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
        
        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.to(self.device)
        self.net.eval()
        rospy.loginfo("Model loaded successfully!")
        
        # Image preprocessing (same as training)
        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # ROS subscribers and publishers
        camera_topic = rospy.get_param('~camera_topic', '/carla/ego_vehicle/rgb_front/image')
        self.image_sub = rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1)
        self.result_pub = rospy.Publisher('/lane_detection/result_image', Image, queue_size=1)
        
        rospy.loginfo(f"Subscribed to: {camera_topic}")
        rospy.loginfo(f"Publishing to: /lane_detection/result_image")
        rospy.loginfo("Lane detection node ready!")

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_h, img_w = cv_image.shape[:2]
            
            # Preprocess image for model
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            img_tensor = self.img_transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.net(img_tensor)
            
            # Process model output
            col_sample = np.linspace(0, 800 - 1, self.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]
            
            out_j = output[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            
            # Apply softmax to get lane positions
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(self.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j_argmax = np.argmax(out_j, axis=0)
            loc[out_j_argmax == self.griding_num] = 0
            out_j = loc
            
            # Visualize detected lanes on image
            result_img = cv_image.copy()
            
            # Different colors for each lane
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR format
            
            for lane_idx in range(out_j.shape[1]):
                if np.sum(out_j[:, lane_idx] != 0) > 2:  # Only draw if lane has enough points
                    color = colors[lane_idx % len(colors)]
                    lane_points = []
                    
                    for k in range(out_j.shape[0]):
                        if out_j[k, lane_idx] > 0:
                            # Map coordinates from model space to image space
                            # First to CULane dimensions, then scale to actual image
                            x_culane = int(out_j[k, lane_idx] * col_sample_w * self.culane_img_w / 800) - 1
                            y_culane = int(self.culane_img_h * (self.culane_row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1
                            
                            # Scale to actual CARLA image dimensions
                            x = int(x_culane * img_w / self.culane_img_w)
                            y = int(y_culane * img_h / self.culane_img_h)
                            
                            # Ensure coordinates are within image bounds
                            x = max(0, min(x, img_w - 1))
                            y = max(0, min(y, img_h - 1))
                            
                            lane_points.append((x, y))
                            cv2.circle(result_img, (x, y), 4, color, -1)
                    
                    # Draw connecting lines between points
                    if len(lane_points) > 1:
                        for j in range(len(lane_points) - 1):
                            cv2.line(result_img, lane_points[j], lane_points[j+1], color, 2)
            
            # Publish result image
            result_msg = self.bridge.cv2_to_imgmsg(result_img, "bgr8")
            result_msg.header = msg.header  # Keep original timestamp
            self.result_pub.publish(result_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = CarlaLaneDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Lane detection node shutting down")
