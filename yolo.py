# Thomas Manteaux (thomas.manteaux@epfl.ch) - July 2023
# Script to detect water bottles in an image. It gives the distance and angle from the camera to teh bottles detected
# All details about camera calibration can be found here https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import cv2 as cv
import numpy as np
import calibration
from datetime import datetime
from time import time
import torch

# Distance parameters
FOCAL_LENGTH = 3.04 #mm
OBJECT_HEIGHT = 195 #mm #Bottle dimensions
OBJECT_WIDTH  = 58 #mm  #Bottle dimensions
SENSOR_HEIGHT = 2.76 #mm, IMX219 for RPi Camera V2
SENSOR_WIDTH = 3.68 #mm, IMX219 for RPi Camera V2
PX_SENSOR_WIDTH = 3280 #px
PX_SENSOR_HEIGHT = 2464 #px
SENSOR_DIAG = np.sqrt(SENSOR_HEIGHT**2 + SENSOR_WIDTH**2)
OBJECT_DIAG =  np.sqrt(OBJECT_HEIGHT**2 + OBJECT_WIDTH**2)
HFOV = 62.2 #deg #RPi camera V2

class Yolo:

    # initialize 
    def __init__(self, weights_path):

        self.model = torch.load()

        #YOLO stuff
        self.net = cv.dnn.readNet("/trained_model/weights/best.pt")
        # self.net = cv.dnn.readNet("/home/pi/test_thomas/iac-monorepo/src/object_detection/src/YOLO_COCO/yolov3_copy.weights", "/home/pi/test_thomas/iac-monorepo/src/object_detection/src/YOLO_COCO/yolov3.cfg")
        self.classes = []
        with open("/home/pi/test_thomas/iac-monorepo/src/object_detection/src/YOLO_COCO/coco.names","r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # calibration attributes
        self.width = 0
        self.height = 0
        self.diag = 0
        self.channel = 0
        self.mtx = np.array((3,3))
        self.newcameramtx = np.array((3,3))
        self.roi = 0
        self.dist = 0

    # calibration of the camera
    def camera_calibrate(self):

        calibration.calibrate()

    # get parameters of the image
    def set_img(self, img):
        # undistort
        dst, self.newcameramtx = calibration.undistort(img)

        self.height, self.width, self.channel = img.shape
        self.diag = np.hypot(self.height, self.width)
        return img

    # check the reprojection error from the calibration step
    def calibration_check(self):
        error = calibration.check_error()
        assert error<0.1, "bad calibration"

    # run YOLO network
    def detection(self, img):
        # Detecting Objects
        blob = cv.dnn.blobFromImage(img, 1/255, (160,160), (0, 0, 0), swapRB=True, crop=False) #size is (160,160) to reduce the computation time

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Show information on the image
        # Initialise variables
        class_ids = []
        confidences = [] 
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                #if confidence>0.5 and str(classes[class_id]) == "bottle":
                if str(self.classes[class_id]) == "bottle": # add some confidence ?
                    # Object detected
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width) 
                    h = int(detection[3] * self.height) # '-10' to check with other detection. In order to make the rectangle fit the edges of the bottle

                    # Rectangle coordinates
                    x = int (center_x - w/2)
                    y = int (center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non Maximum Suppresion (Keep one box)
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        objects = []
        # Label object
        font = cv.FONT_HERSHEY_PLAIN
        img = self.crop(img)  #crop to reduce computation time
        img = cv.pyrDown(img) #downsample to reduce computation time
        for i in range (len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                objects.append([x, y, w, h])
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = self.colors[i]

                cv.rectangle(img, (int(x/2),int((y-80)/2)), (int((x+w)/2), int((y-80+h)/2)), color, 2)
                cv.putText(img, label + "  " + confidence, (x, y+20), font, 1, color, 2)

                cv.imwrite('/home/pi/test_thomas/iac-monorepo/src/object_detection/src/images/' + str(datetime.now()) + '.jpg', img)

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
    
    # crop ROI
    def crop(self, img):
        y=80
        x=0
        h=500
        w=1000
        return img[y:y+h, x:x+w]
    
def run():
    YOLO = Yolo()

    # Load Image
    img = cv.imread('/home/pi/test_thomas/iac-monorepo/src/object_detection/src/images/presentation/img2023-07-17 12:10:11.534740.jpg')
    YOLO.camera_calibrate()
    
    img = YOLO.set_img(img)
    #check calibration
    YOLO.calibration_check()

    #object detection
    temps = time()
    object_list = YOLO.detection(img)
    print('time', time()-temps)
    #print(object_list)

    #Distance & angle computation
    #Ensures correct values for object located < 0.8m from the camera
    angle_rad = YOLO.angle(object_list)
    distances = YOLO.distance(object_list, angle_rad)
    for i in range(len(object_list)):
        print('distances (m)', distances[i], 'angle', np.round(angle_rad[i]*180/np.pi,1))

    object_coordinates = YOLO.obstacle_coordinates(1, 2, 30, distances, angle_rad)
    obstacle_list = np.column_stack((object_coordinates, [np.round(a[2]/1000,3) for a in object_list], np.ones(len(object_list))))

    print(obstacle_list)

#Run main function         
if __name__ == '__main__':
    run()
