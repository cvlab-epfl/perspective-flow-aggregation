import numpy as np
import pyrender
import cv2

def render_objects(meshes, ids, poses, K, w, h):
    '''
    '''
    assert(K[0][1] == 0 and K[1][0] == 0 and K[2][0] ==0 and K[2][1] == 0 and K[2][2] == 1)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    objCnt = len(ids)
    assert(len(poses) == objCnt)
    
    # set background with 0 alpha, important for RGBA rendering
    scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 0.0]), ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    # pyrender.Viewer(scene, use_raymond_lighting=True)
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera = pyrender.IntrinsicsCamera(fx=fx,fy=fy,cx=cx,cy=cy,znear=0.05,zfar=100000)
    camera_pose = np.eye(4)
    # reverse the direction of Y and Z, check: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    camera_pose[1][1] = -1
    camera_pose[2][2] = -1
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=4.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    # light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=camera_pose)
    for i in range(objCnt):
        clsId = int(ids[i])
        mesh = pyrender.Mesh.from_trimesh(meshes[clsId])

        H = np.zeros((4,4))
        H[0:3] = poses[i][0:3]
        H[3][3] = 1.0
        scene.add(mesh, pose=H)

    # pyrender.Viewer(scene, use_raymond_lighting=True)

    r = pyrender.OffscreenRenderer(w, h)
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
    # flags = pyrender.RenderFlags.OFFSCREEN
    flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
    color, depth = r.render(scene, flags=flags)
    # color, depth = r.render(scene)
    # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) # RGB to BGR (for OpenCV)
    color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA) # RGBA to BGRA (for OpenCV)
    # # 
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(color)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    # # plt.imshow(depth)
    # plt.show()
    # # 
    # r.delete()
    # color = None
    # 
    return color, depth

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
