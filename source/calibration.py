import numpy as np
import cv2
import glob


#Загрузка всех картинок из директории
def load_img():
    cv_img = []
    for img in glob.glob("C:/Users/tonym/Desktop/ProjectX/calib_extra/*.jpg"):
        n= cv2.imread(img)
        cv_img.append(n)
    return cv_img

#Критерий остановки
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Подготовка точек обекта: (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objpoints = [] # 3d точки из реального мира
imgpoints = [] # 2d точки изображения поверхности 
images = load_img()
cc= 0

for fname in images:
    gray = cv2.cvtColor(fname,cv2.COLOR_RGB2GRAY)

    #Поиск углов шахматной доски 
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    #Если углы найдены
    if ret == True:
        cc += 1
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        cv2.waitKey(500)

#Калибровка камеры            
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx)
print(cc)
#Для камеры 0
'''
[[862.44357443   0.         261.31601578]
 [  0.         859.96288926 176.84476175]
 [  0.           0.           1.        ]]
'''
#Для камеры 1
'''
[[693.90888662   0.         333.27294343]
 [  0.         688.54241995 237.91849834]
 [  0.           0.           1.        ]]
'''


