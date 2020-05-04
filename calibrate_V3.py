# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:20:28 2020

@author: srussell
"""

import cv2
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
print(cv2.__version__)
import numpy as np
import os
import glob
import sys
import json
# OPENCV_PYTHON_DEBUG=1

def detect_charts (file_exp="", output_path = ""):

    images = glob.glob(file_exp)

    data = {}
    data['detected_corners'] = []
    _img_shape = None
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        print(img.shape)
        if (_img_shape == None):
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)#np.append(objpoints,objp, axis = 0)
            # print(objp)
            # print(corners)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)#np.append(imgpoints,corners, axis = 1)

            data['detected_corners'].append({
                'image_shape' : _img_shape,
                'image_path' : fname,    
                'objpoints' : objp.tolist(),
                'imgpoints' : corners.tolist() 
            })
    if output_path != "":
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile)
    else:
        data = {}


def Read_Json(file=''):
    # #read from JSON file
    objpoints = []
    imgpoints = []
    _img_shape = []
    with open(file) as json_file:
        data = json.load(json_file)
        for p in data['detected_corners']:
            _img_shape = p['image_shape']
            objpoints.append(np.array(p['objpoints'],dtype = np.float32))
            imgpoints.append(np.array(p['imgpoints'],dtype = np.float32))

    return objpoints,imgpoints,_img_shape



#Performing camera calibration by
#passing the value of known 3D points (objpoints)
#and corresponding pixel coordinates of the
#detected corners (imgpoints)
  
  
# Pinhole      
def Pinhole_dist(image_in, image_out):
    print("\nPINHOLE \n")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (_img_shape[::-1][0],_img_shape[::-1][1]), None, None)
    print("\nCamera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    #projectPoints
    for index,points in enumerate(imgpoints):
        imgpoints_reprojection,_ = cv2.projectPoints(objpoints[index],rvecs[index],tvecs[index],mtx,dist)
        print(points)
        # imgpoints_reprojection_error = points - imgpoints_reprojection
        # # print('error: ' + str(imgpoints_reprojection_error ))
        # imgpoints_reprojection_max = np.max(np.ravel(imgpoints_reprojection_error))
        # print('max: ' + str(imgpoints_reprojection_max ))
        # imgpoints_reprojection_min = np.min(np.ravel(imgpoints_reprojection_error))
        # print('min: ' + str(imgpoints_reprojection_min ))
        # imgpoints_reprojection_avg = np.average(np.ravel(imgpoints_reprojection_error))
        # print('avg: ' + str(imgpoints_reprojection_avg ))
        # imgpoints_reprojection_std = np.std(np.ravel(imgpoints_reprojection_error))
        # print('std: ' + str(imgpoints_reprojection_std ))
        print('------')

    img = cv2.imread(image_in)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(image_out,dst)

###Fisheye
def Fisheye_dist(image_in='', image_out=''):
    print("\n Fisheye \n")
    N_OK = len(objpoints)
    # K = np.zeros((3, 3))
    # D = np.zeros((4, 1))
    # rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    # tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, K, D, rvecs, tvecs = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            (_img_shape[::-1][0],_img_shape[::-1][1]),
            K = None,
            D = None,
            rvecs=None,
            tvecs=None,
            flags = calibration_flags,
            criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    # imageFile = r'./original/capture_87.jpg'
    projected =[]
    imgpoints_reprojection1=[]
    points1=[]
    # OriImage = cv2.imread(imageFile)
    #projectPoints
    print('---reprojection error---')
    for index,points in enumerate(imgpoints):
        points1.append(points.tolist())
        imgpoints_reprojection,_ = cv2.fisheye.projectPoints(objpoints[index],rvecs[index],tvecs[index],K,D)
        imgpoints_reprojection1.append({'projected': imgpoints_reprojection.tolist()})
       # for p in imgpoints_reprojection1['projected']:
        #    projected.append(np.array(p['projected'],dtype = np.float32))
        # for i in enumerate(points):
        #     projected[i]=points[i]
       # c = list(np.array(points1) - np.array(imgpoints_reprojection1))
        #projected.append(np.array([imgpoints_reprojection1],dtype = np.float32))
        print(imgpoints_reprojection)
        # for i in range(points.size):
        #     projected[i+1] = points[i+1]-imgpoints_reprojection[i+1]
        #     print(projected[i+1])
        # points1 =points.tolist()
        # projected=imgpoints_reprojection.tolist()
        # print(imgpoints_reprojection-points)
        # # imgpoints_reprojection_error = points - imgpoints_reprojection
        # # print('error: ' + str(imgpoints_reprojection_error ))
        # # imgpoints_reprojection_max = np.max(np.ravel(imgpoints_reprojection_error))
        # # print('max: ' + str(imgpoints_reprojection_max ))
        # # imgpoints_reprojection_min = np.min(np.ravel(imgpoints_reprojection_error))
        # # print('min: ' + str(imgpoints_reprojection_min ))
        # # imgpoints_reprojection_avg = np.average(np.ravel(imgpoints_reprojection_error))
        # # print('avg: ' + str(imgpoints_reprojection_avg ))
        # # imgpoints_reprojection_std = np.std(np.ravel(imgpoints_reprojection_error))
        # # print('std: ' + str(imgpoints_reprojection_std ))
        # print('------')

    #DIM=" + str(_img_shape[::-1])

    #DIM=(2592, 1944)
    #K=np.array([[154.98364190838828, 0.0, 157.67109328055724], [0.0, 154.6314452815455, 106.8013597157768], [0.0, 0.0, 1.0]])
    K=np.array([K[0], K[1], K[2]])
    D=np.array([D[0], D[1], D[2], D[3]])

    img = cv2.imread(image_in)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (_img_shape[::-1][0],_img_shape[::-1][1]), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(image_out, undistorted_img)
    #cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
# if __name__ == '__main__':
#     for p in sys.argv[1:]:
#         undistort(p)
       

CHECKERBOARD = (7,11)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []#np.empty([1,77,3], dtype=np.float32) # 3d point in real world space
imgpoints = []#np.empty([77,1,2], dtype=np.float32) # 2d points in image plane.
_img_shape = None

if __name__ == '__main__':
    
    # img_in = './check_board_capture/check_board_small/selected/output/capture_04.jpg'
    # img_out = './check_board_capture/check_board_small/selected/output/capture_04_ouput.jpg'
    #detect_charts("./fisheye/*.jpg", "./fisheye/data_fisheye.json")

    objpoints,imgpoints,_img_shape = Read_Json('./fisheye/data_fisheye.json')

    images_in = glob.glob('./original/*.jpg')
    for img_in in images_in:
        Fisheye_dist(img_in,'./fisheye_fisheye/'+ os.path.basename(img_in))

    # for img_in in images_in:
    #     Pinhole_dist(img_in,'./fisheye_pinhole/'+ os.path.basename(img_in))
    
    # Pinhole_dist(img_in,'./check_board_capture/check_board_small/selected/output/capture_04_ouput_pin.jpg')

