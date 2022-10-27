import numpy as np
import cv2


def draw_bounding_box(cvImg, R, T, mesh, intrinsics, color):
    thickness = 2
    bbox = mesh.bounding_box_oriented.vertices
    rep = np.matmul(intrinsics, np.matmul(R, bbox.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    bbox_lines = [0, 1, 0, 2, 0, 4, 5, 1, 5, 4, 6, 2, 6, 4, 3, 2, 3, 1, 7, 3, 7, 5, 7, 6]
    for i in range(12):
        id1 = bbox_lines[2*i]
        id2 = bbox_lines[2*i+1]
        cvImg = cv2.line(cvImg, (x[id1],y[id1]), (x[id2],y[id2]), color, thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_pose_axis(cvImg, R, T, mesh, intrinsics):
    thickness = 2
    bbox = mesh.bounding_box_oriented.vertices
    radius = np.linalg.norm(bbox, axis=1).mean()
    aPts = np.array([[0,0,0],[0,0,radius],[0,radius,0],[radius,0,0]])
    rep = np.matmul(intrinsics, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[1],y[1]), (0,0,255), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[2],y[2]), (0,255,0), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[3],y[3]), (255,0,0), thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_z_axis(cvImg, R, T, mesh, intrinsics, color):
    thickness = 2
    bbox = mesh.bounding_box_oriented.vertices
    radius = np.linalg.norm(bbox, axis=1).mean()
    aPts = np.array([[0,0,0],[0,0,radius]])
    rep = np.matmul(intrinsics, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[1],y[1]), color, thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_pose_contour(cvImg, R, T, mesh, K, color):
    # 
    h, w, _ = cvImg.shape
    currentpose = np.concatenate((R, T.reshape(-1, 1)), axis=-1)
    _, depth = render_objects([mesh], [0], [currentpose], K, w, h)
    validMap = (depth>0).astype(np.uint8)
    # 
    # find contour
    contours, _ = cv2.findContours(validMap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # cvImg = cv2.drawContours(cvImg, contours, -1, (255, 255, 255), 1, cv2.LINE_AA) # border
    cvImg = cv2.drawContours(cvImg, contours, -1, color, 2)
    return cvImg
