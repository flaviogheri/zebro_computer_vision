import cv2 as cv
import numpy as np
import torch
from datetime import datetime
import calibration
from time import time
# from yolov5 import detect


###################### TO BE VERIFIED ################
OBJECT_HEIGHT = 120 #mm, Polygon dimensions
OBJECT_WIDTH  = 120 #mm, Polygon Dimensions



# Sensor parameters - Imported from THOMAS (need to be verified)

FOCAL_LENGTH = 3.04 #mm

SENSOR_HEIGHT = 2.76 #mm, IMX219 for RPi Camera V2
SENSOR_WIDTH = 3.68 #mm, IMX219 for RPi Camera V2
PX_SENSOR_WIDTH = 3280 #px
PX_SENSOR_HEIGHT = 2464 #px
SENSOR_DIAG = np.sqrt(SENSOR_HEIGHT**2 + SENSOR_WIDTH**2)
OBJECT_DIAG =  np.sqrt(OBJECT_HEIGHT**2 + OBJECT_WIDTH**2)
HFOV = 62.2 #deg #RPi camera V2



class Yolo: 

    def __init__(self):

        self.weights_path = 'trained_model/weights/best.pt'
        self.model = torch.hub.load('yolov5_original', 'custom', path=self.weights_path, source='local')
        # self.model = torch.load(self.weights_path)['model'].float()
        self.model.eval()
        self.classes = ['polygon']

        # # calibration attributes
        # self.width = 0
        # self.height = 0
        # self.diag = 0
        # self.channel = 0
        # self.mtx = np.array((3,3))
        # self.newcameramtx = np.array((3,3))
        # self.roi = 0
        # self.dist = 0

    # extract camera calibration matrix
    def camera_calibrate(self):
        calibration.calibrate()
    
    # def crop_image(image, size=(160,160)):
    #     h, w = image.shape[:2]
    #     start_h = (h - size[1]) // 2
    #     start_w = (w - size[0]) // 2
    #     cropped_image = image[start_h:start_h+size[1], start_w:start_w+size[0]]
    #     return cropped_image

    # (temporarily use Thomas method instead)
    def crop(self, img):
        y=80
        x=0
        h=500
        w=1000
        return img[y:y+h, x:x+w]

        
    # calibrates the image and retrieves param.    
    def set_img(self, img):

        dst, self.newcameramtx = calibration.undistort(img)

        self.height, self.width, self.channel = img.shape
        self.diag = np.hypot(self.height, self.width)
        return img
    
    # check the reprojection error from the calibration step
    def calibration_check(self):
        error = calibration.check_error()
        assert error<0.1, "bad calibration"

    def detection(self, img):
        
        # temporarily use thomas crop for now
        # img = self.crop_image(img)
        img = self.crop(img)
        results = self.model(img)
        objects = []

        # read if doesnt work: https://github.com/ultralytics/yolov5/issues/36

        for det in results.xyxy[0]: 
            x1, y1, x2, y2, conf, cls = det
            if self.classes[int(cls)] == 'polygon' and conf > 0.6:
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                objects.append([x, y, w, h])

        #         # for box visualization
        #         color = (0, 255, 0)
        #         cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #         cv.putText(img, f'{self.classes[int(cls)]} {conf:.2f}', (x, y - 10), cv.FONT_HERSHEY_PLAIN, 1, color, 2)
        #     # Display the image with drawn bounding boxes
        # cv.imshow('Detected Objects', img)
        # while True: 
        #     key = cv.waitKey(1)  # Wait indefinitely for a key press
        #     if key == ord('q') or cv.getWindowProperty('Detected Objects', cv.WND_PROP_VISIBLE) < 1:
        #         break      
        return objects

    # get the distance between the objects and the camera
    def distance(self, obj_list, angle_rad):
        dist = []
        for i in range(len(obj_list)):
            diag = np.sqrt(obj_list[i][2]**2 + obj_list[i][3]**2)
            distances = (FOCAL_LENGTH*self.diag*OBJECT_DIAG) / (diag*SENSOR_DIAG*1000) + 0.1
            dist.append(np.round(distances/np.cos(angle_rad[i]),2))

        return dist #meters
    
     # get the angle between the objects and the camera
    def angle(self, obj_list):

        K = np.array([[2710, 0, PX_SENSOR_WIDTH/2], [0, 2710, PX_SENSOR_HEIGHT/2], [0, 0, 1]])
        #K = self.newcameramtx #from calibration

        angle_radians = []
        for i in range(len(obj_list)):

            P = np.array([obj_list[i][0]+obj_list[i][2]/2, obj_list[i][1]+obj_list[i][3]/2 ,1.0])
            p = K.dot(P)
            p = P
            x, y = p[0]/p[2], p[1]/p[2]

            ecart = np.abs(self.width/2 - x)

            if x < self.width/2:
                ang = -(ecart*HFOV/self.width)*np.pi/180
            else:
                ang = (ecart*HFOV/self.width)*np.pi/180

            angle_radians.append(ang)

        return angle_radians

    # define absolute coordinates of the obstacles based on the camera position
    def obstacle_coordinates(self, x_r, y_r, heading, dist, angles):

        coordinates = []
        for i in range(len(dist)):
            absolute_angle = heading*np.pi/180 + angles[i]
            x_o = np.round(x_r + dist[i]*np.sin(absolute_angle),2)
            y_o = np.round(y_r + dist[i]*np.cos(absolute_angle),2)
            coordinates.append([x_o,y_o])
        
        #print('coordinates', coordinates)
        return coordinates

def run():
    YOLO = Yolo()

    # Take Picture
    cap = cv.VideoCapture(0)
    
    ret, img = cap.read()


    # Calibrate Image
    YOLO.camera_calibrate() # finds calibration matrix
    img = YOLO.set_img(img) # calibrates image
    YOLO.calibration_check() # checks calibration is correct

    # Object Detection
    temps = time()
    object_list = YOLO.detection(img)
    print('time', time()-temps)
    
    # Distance Computation
    if object_list != None:
        angle_rad = YOLO.angle(object_list)
        distances = YOLO.distance(object_list, angle_rad)
        for i in range(len(object_list)):
            print('distances (m)', distances[i], 'angle', np.round(angle_rad[i]*180/np.pi,1))

        object_coordinates = YOLO.obstacle_coordinates(1, 2, 30, distances, angle_rad)
        obstacle_list = np.column_stack((object_coordinates, [np.round(a[2]/1000,3) for a in object_list], np.ones(len(object_list))))
        print(obstacle_list)
    else: 
        print("error obj list is nonetype")
if __name__ == '__main__':
    run()

    ############# THINGS TO DO: 

    """
    1. FINISH DISTANCE MEASUREMENT

    2. FINISH ALL CLASS FUNCTIONS

    3. RUN A TEST

    4. PUSH TO GITLAB

    5. MAKE SURE IT WORKS ON ROBOT

    6. GET UNIT TESTS/EASIER METHOD OF interchanging models for future integration
    
    """