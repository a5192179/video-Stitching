import cv2
import numpy as np
import glob
import pickle

def cal_real_corner(corner_height, corner_width, square_size):
    obj_corner = np.zeros([corner_height * corner_width, 3], np.float32)
    obj_corner[:, :2] = np.mgrid[0:corner_height, 0:corner_width].T.reshape(-1, 2)  # (w*h)*2
    return obj_corner * square_size

def calibration(fileNameTemp, corner_height:int, corner_width:int, square_size:float):
    '''
    fileNameTemp = "D:/project/videoFusion/3rdparty/camera_calibration_tool/chess/*.jpg"
    corner_height: 高度方向角点的数量（不是方块的数量）
    corner_width: 宽度方向角点的数量（不是方块的数量）
    square_size：方块宽度(mm)
    '''
    # file_names = glob.glob('./chess/*.JPG') + glob.glob('./chess/*.jpg') + glob.glob('./chess/*.png')
    
    file_names = glob.glob(fileNameTemp)
    objs_corner = []
    imgs_corner = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_corner = cal_real_corner(corner_height, corner_width, square_size)
    for file_name in file_names:
        # read image
        chess_img = cv2.imread(file_name)
        image_size = tuple([chess_img.shape[1], chess_img.shape[0]])
        assert (chess_img.shape[0] == image_size[1] and chess_img.shape[1] == image_size[0]), \
            "Image size does not match the given value {}.".format(image_size)
        # to gray
        gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
        # find chessboard corners
        ret, img_corners = cv2.findChessboardCorners(gray, (corner_height, corner_width))

        # append to img_corners
        if ret:
            objs_corner.append(obj_corner)
            img_corners = cv2.cornerSubPix(gray, img_corners, winSize=(square_size//2, square_size//2),
                                            zeroZone=(-1, -1), criteria=criteria)
            imgs_corner.append(img_corners)
        else:
            print("Fail to find corners in {}.".format(file_name))

    # calibration
    ret, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objs_corner, imgs_corner, image_size, None, None)
    # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, image_size, alpha=1)
    # roi = np.array(roi)
    print("ret:", ret)
    print("mtx:\n", matrix) # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs ) # 平移向量  # 外参数
    # img = cv2.undistort(chess_img, matrix, dist)
    # img = cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6)))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return ret, matrix, dist, rvecs, tvecs

if __name__ == "__main__":
    caliImgFolder = '../data/Photo/cali'
    cameraID = '132'
    fileNameTemp = caliImgFolder + '/*.jpg'
    corner_height = 6
    corner_width = 9
    square_size = 30
    ret, matrix, dist, rvecs, tveces = calibration(fileNameTemp, corner_height, corner_width, square_size)
    saveFile = caliImgFolder + '/cameraMatrix' + cameraID + '.txt'
    f = open(saveFile, "wb")
    f.write(pickle.dumps(matrix))
    f.close()
    saveFile = caliImgFolder + '/cameraDist' + cameraID + '.txt'
    f = open(saveFile, "wb")
    f.write(pickle.dumps(dist))
    f.close()


# # calibration(6, 9, 30)

# images = glob.glob("../data/Photo/cali/*.jpg")
# # images = glob.glob("D:/project/videoFusion/3rdparty/camera_calibration_tool/chess/*.jpg")
# corner_u = 9
# corner_v = 6
# # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
# # criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
# criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)


# # 获取标定板角点的位置
# objp = np.zeros((corner_v * corner_u, 3), np.float32)
# objp[:, :2] = np.mgrid[0:corner_v, 0:corner_u].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
# objp = objp * 30

# obj_points = []  # 存储3D点
# img_points = []  # 存储2D点


# for fname in images:
#     img = cv2.imread(fname)
#     cv2.imshow('img',img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # size = gray.shape[::-1]
#     size = tuple([gray.shape[1], gray.shape[0]])
#     # ret, corners = cv2.findChessboardCorners(gray, (corner_v, corner_u), None)
#     ret, corners = cv2.findChessboardCorners(gray, (corner_v, corner_u))
#     print(ret)

#     if ret:

#         obj_points.append(objp)

#         corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
#         print('corners2', corners2)
#         if [corners2]:
#             img_points.append(corners2)
#         else:
#             img_points.append(corners)

#         cv2.drawChessboardCorners(img, (corner_v, corner_u), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
#         # cv2.imshow('img', img)
#         # cv2.waitKey(2000)

# print(len(img_points))
# cv2.destroyAllWindows()

# # 标定
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
# # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, size, alpha=1)
# print("ret:", ret)
# print("mtx:\n", mtx) # 内参数矩阵
# print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
# print("tvecs:\n", tvecs ) # 平移向量  # 外参数

# print("-----------------------------------------------------")

# # test cali
# img = cv2.imread('D:/project/videoFusion/data/2020122901/img/0.jpg')
# imgUn = cv2.undistort(src = img, cameraMatrix = mtx, distCoeffs = dist)
# cv2.imshow('img', imgUn)
# cv2.waitKey(0)
# a=1