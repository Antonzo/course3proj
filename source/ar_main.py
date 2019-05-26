import cv2
import numpy as np
import math
from objloader import OBJ
import random 
MIN_MATCHES = 12

def main():
    homography = None 
    # для камеры 1
    camera_parameters = np.array([[693.90888662, 0, 333.27294343], [0, 688.54241995, 237.91849834], [0, 0, 1]])
    # для камеры 0
    #camera_parameters = np.array([[862.44357443, 0, 261.31601578], [0, 859.96288926, 176.84476175], [0, 0, 1]])
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    model = cv2.imread("C:/Users/tonym/Desktop/ProjectX/reference/model7.jpg", 0)
    kp_model = orb.detect(model)
    kp_model, des_model = orb.compute(model, kp_model)
    obj = OBJ("C:/Users/tonym/Desktop/ProjectX/models/rat.obj", swapyz=True)  
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
           
        if cv2.waitKey(1) == ord('q'):
            break
        ret, scene = cap.read()
        if not ret:
            print("Unable to capture video")
            return 
        kp_scene = orb.detect(scene)
        kp_scene, des_scene = orb.compute(scene, kp_scene)
        matches = bf.match(des_model, des_scene)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > MIN_MATCHES:
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is not None:
                try:    
                    projection = projection_matrix(camera_parameters, homography)  
                    scene = render(scene, obj, projection, model)
                except:
                    pass
                
            cv2.imshow('frame', scene)
            cv2.imwrite("C:/Users/tonym/Desktop/ProjectX/rat_testing/img" + str(random.randint(1, 10000)) + ".jpg", scene)
        else:
            print("Недостаточно совпадений - %d/%d" % (len(matches), MIN_MATCHES))
    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model):
    vertices = obj.vertices
    scale = 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = points * scale
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)       
        cv2.fillConvexPoly(img, imgpts, (110, 110, 110))
    return img

def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    g_1 = rot_and_transl[:, 0]
    g_2 = rot_and_transl[:, 1]
    g_3 = rot_and_transl[:, 2]
    l = math.sqrt(np.linalg.norm(g_1, 2) * np.linalg.norm(g_2, 2))
    r_1 = g_1 / l
    r_2 = g_2 / l
    t = g_3 / l
    e_1 = r_1 + r_2
    e_2 = np.cross(r_1 + r_2, np.cross(r_1, r_2))
    r_1 = np.dot(e_1 / np.linalg.norm(e_1, 2) + e_2 / np.linalg.norm(e_2, 2), 1 / math.sqrt(2))
    r_2 = np.dot(e_1 / np.linalg.norm(e_1, 2) - e_2 / np.linalg.norm(e_2, 2), 1 / math.sqrt(2))
    r_3 = np.cross(r_1, r_2)
    projection = np.stack((r_1, r_2, r_3, t)).T
    return np.dot(camera_parameters, projection)


if __name__ == '__main__':
    main()
