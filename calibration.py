import numpy as np
import cv2 as cv
import glob

## Setup ##    

## Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def calibrate():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

    images = glob.glob('calibration/*.jpg')

    for fname in images:
        img = cv.imread(fname)
        #print(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        global ret
        ret, corners = cv.findChessboardCorners(gray, (9,7), None)
        #print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            #cv.drawChessboardCorners(img, (9,7), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(10)

    #cv.destroyAllWindows()

    ## Calibration ##
    print('calibration done')
    global mtx, dist, rvecs, tvecs
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def undistort(img):
    ## Undistorsion ##
    #img = cv.imread('images/calibration/calibration2.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    #print(dst.shape[:2])
    #cv.imwrite('images/calibration/calibresult.png', dst)

    return dst, newcameramtx

## Re-projection error ##
def check_error():
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    return mean_error/len(objpoints)
