import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')
from common import readInput
import datetime
import time
import math

def detect(image):
    # 转化为灰度图
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建SIFT生成器
    # descriptor是一个对象，这里使用的是SIFT算法
    descriptor = cv2.SIFT_create()
    # 检测特征点及其描述子（128维向量）
    kps, features = descriptor.detectAndCompute(gray, None)
    # kps, features = descriptor.detectAndCompute(image, None)
    print(f"特征点数：{len(kps)}")
    return (kps,features)

def show_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.SIFT_create()
    kps, features = descriptor.detectAndCompute(gray, None)
    # kps, features = descriptor.detectAndCompute(image, None)
    print(f"特征点数：{len(kps)}")
    img_left_points = cv2.drawKeypoints(image, kps, image)
    # plt.figure(figsize=(9,9)) 
    # plt.imshow(img_left_points)
    
def match_keypoints(kps_left,kps_right,features_left,features_right,ratio,threshold):
    """
    kpsA,kpsB,featureA,featureB: 两张图的特征点坐标及特征向量
    threshold: 阀值
    
    """
    # 建立暴力匹配器
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # 使用knn检测，匹配left,right图的特征点
    raw_matches = matcher.knnMatch(features_left, features_right, 2)
    print(len(raw_matches))
    matches = []  # 存坐标，为了后面
    good = [] # 存对象，为了后面的演示
    # 筛选匹配点
    for m in raw_matches:
        # 筛选条件
#         print(m[0].distance,m[1].distance)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            good.append([m[0]])
            matches.append((m[0].queryIdx, m[0].trainIdx))
            """
            queryIdx：测试图像的特征点描述符的下标==>img_keft
            trainIdx：样本图像的特征点描述符下标==>img_right
            distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
            """
    # 特征点对数大于4就够用来构建变换矩阵了
    kps_left = np.float32([kp.pt for kp in kps_left])
    kps_right = np.float32([kp.pt for kp in kps_right])
    print(len(matches))
    if len(matches) > 4:
        # 获取匹配点坐标
        pts_left = np.float32([kps_left[i] for (i,_) in matches])
        pts_right = np.float32([kps_right[i] for (_,i) in matches])
        # 计算变换矩阵(采用ransac算法从pts中选择一部分点)
        H,status = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, threshold)
        return (matches, H, good)
    return None

# def drawMatches(img_left, img_right, kps_left, kps_right, matches, H):
#     # 获取图片宽度和高度
#     h_left, w_left = img_left.shape[:2]
#     h_right, w_right = img_right.shape[:2]
#     """对imgB进行透视变换
#     由于透视变换会改变图片场景的大小，导致部分图片内容看不到
#     所以对图片进行扩展:高度取最高的，宽度为两者相加"""
#     image = np.zeros((max(h_left, h_right), w_left+w_right, 3), dtype='uint8')
#     # 初始化
#     image[0:h_left, 0:w_left] = img_right
#     """利用以获得的单应性矩阵进行变透视换"""
#     image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))#(w,h
#     """将透视变换后的图片与另一张图片进行拼接"""
#     image[0:h_left, 0:w_left] = img_left
#     return image

def remove_black_edges(img):
    u=img.shape[1]
    v=img.shape[0]
    edges_x=[]
    edges_y=[]
    #optimization...

    left = u              #左边界
    right = 0              #右边界
    bottom = 0            #底部
    top = v                #顶部

    for i in range(u):
        for j in range(v):
            if np.sum(img[j][i][:]) != 0:
                if i < left:
                    left = i
                if i > right:
                    right = i
                if j > bottom:
                    bottom = j
                if j < top:
                    top = j
    #...optimization
    # left = min(edges_x)               #左边界
    # right = max(edges_x)              #右边界
    # bottom = min(edges_y)             #底部
    # top = max(edges_y)                #顶部
    imgOut = img[top:bottom, left:right, :]
    cv2.imshow('img', img)
    cv2.imshow('imgOut', imgOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.figure(figsize=(5,5))
    # plt.imshow(img)
    # plt.show()
    # plt.figure(figsize=(5,5))
    # plt.imshow(imgOut)
    # plt.show()
    return imgOut
class ImgFusioner:
    '''
    fusion img with order
    '''
    def __init__(self, imgNum):
        self.imgNum = imgNum # camera id list
        self.HList = []
        self.topList = []
        self.buttomList = []
        self.leftList = []
        self.rightList = []
        self.count = 0
        self.FirstLoop = True
        self.centerU = -1
        self.centerV = -1
        self.cylinderReady = False
        self.remapIndexX = 0
        self.remapIndexY = 0
        self.center = 0

    def projectToCylinder(self, img, theta, center_x = -1, center_y = -1) :
        '''
        theta is fov, unit is rad
        '''
        blank = np.zeros_like(img)
        if not self.cylinderReady:
            rows = img.shape[0]
            cols = img.shape[1]
            self.remapIndexX = np.zeros([rows, cols], dtype='float32')
            self.remapIndexY = np.zeros([rows, cols], dtype='float32')
            
            f = cols / (2 * math.tan(theta / 2))
            # !!! the original is : f = cols / (2 * math.tan(theta / 8))
            
            if center_x == -1:
                center_x = int(cols / 2)
                center_y = int(rows / 2)
            self.center = (center_x, center_y)

            for  y in range(rows):
                for x in range(cols):
                    theta = math.atan((x- center_x )/ f)
                    point_x = int(f * math.tan( (x-center_x) / f) + center_x)
                    point_y = int( (y-center_y) / math.cos(theta) + center_y)
                    # point_y = y
                    
                    # if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                    #     pass
                    # else:
                    #     blank[y , x, :] = img[point_y , point_x ,:] !!!
                    self.remapIndexX[y, x] = float(point_x)
                    self.remapIndexY[y, x] = float(point_y)
            self.cylinderReady = True
                    
        blank = cv2.remap(img, self.remapIndexX, self.remapIndexY, cv2.INTER_LINEAR)
        radius = 10
        color = (0, 0, 255)
        thickness = 2
        blank = cv2.circle(blank, self.center, radius, color, thickness)
        # cv2.imshow('img', cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6))))
        # cv2.imshow('cylinder', cv2.resize(blank, (int(blank.shape[1] * 0.6), int(blank.shape[0] * 0.6))))
        # cv2.waitKey(0)
        return blank

    def fusionByAffine(self, imgNew, imgTurn):
        '''
        生成一张大图，类似3*3，其中中间是imgNew，四角是imgTurn，其他部位可以根据hNew + 2 * hTurn, wNew + 2 * wTurn推测
        以imgNew为基准，先把imgTurn放到background的 hTurn wTurn （左上角）位置
        用wrap affine 转换 background
        把imgNew放到background
        剔除额外的边界
        '''
        hNew, wNew = imgNew.shape[:2]
        hTurn, wTurn = imgTurn.shape[:2]
        # 1.prepare a big picture
        background = np.zeros((hNew + 2 * hTurn, wNew + 2 * wTurn, 3), dtype='uint8')
        # 2.turn img
        if len(self.HList) < self.count + 1:
            print('count:', self.count, 'generate data')
            background[hTurn:hTurn + hTurn, wTurn:wTurn + wTurn, :] = imgTurn
            temp = np.zeros((hNew + 2 * hTurn, wNew + 2 * wTurn, 3), dtype='uint8') # 对 imgNew 也要进行扩张，计算的H是相对imgNew的形状，如果不扩张，扩张后的imgTurn，原点还是对应扩张前imgNew的原点
            temp[hTurn:hTurn + hNew, wTurn:wTurn + wNew, :] = imgNew
            # 模块一：提取特征
            kpsNew, featuresNew = detect(temp)
            kpsTurn, featuresTurn = detect(background)
            # cv2.imshow('img_left', img_left)
            # cv2.imshow('img_right', img_right)
            # cv2.waitKey(2000)

            # 模块二：特征匹配
            matches, H, good = match_keypoints(kpsNew,kpsTurn,featuresNew,featuresTurn,0.5,0.99)
            print('matches:', len(matches))
            # ============================
            img = cv2.drawMatchesKnn(temp,kpsNew,background,kpsTurn,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('IMG', (cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # plt.figure(figsize=(20,20))
            # plt.imshow(cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6))))
            # ============================
            # 模块三：透视变换-拼接
            self.HList.append(H)
            # cv2.imshow('background0', cv2.resize(background, (int(background.shape[1] * 0.6), int(background.shape[0] * 0.6))))
            background = cv2.warpPerspective(background, H, (background.shape[1], background.shape[0])) #注意，这里的形状参数是（列，行）
            # cv2.imshow('background1', cv2.resize(temp, (int(temp.shape[1] * 0.6), int(temp.shape[0] * 0.6))))
            # cv2.waitKey(0)

            # topLeft 
            u = wTurn
            v = hTurn
            topLeftV = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2]) #https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
            topLeftU = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])

            # topRight
            u = wTurn + wTurn
            v = hTurn
            topRightV = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])
            topRightU = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])

            # buttomLeft 
            u = wTurn
            v = hTurn + hTurn
            buttomLeftV = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])
            buttomLeftU = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])

            # buttomRight 
            u = wTurn + wTurn
            v = hTurn + hTurn
            buttomRightV = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])
            buttomRightU = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])

            # topLeftNew
            topLeftNewV = hTurn
            topLeftNewU = wTurn

            # topRightNew
            topRightNewV = hTurn
            topRightNewU = wTurn + wNew

            # buttomLeftNew
            buttomLeftNewV = hTurn + hNew
            buttomLeftNewU = wTurn

            # buttomRightNew
            buttomRightNewV = hTurn + hNew
            buttomRightNewU = wTurn + wNew

            top = min(topLeftV, topRightV, buttomLeftV, buttomRightV, topLeftNewV, topRightNewV, buttomLeftNewV, buttomRightNewV)
            top = max(0, top)
            buttom = max(topLeftV, topRightV, buttomLeftV, buttomRightV, topLeftNewV, topRightNewV, buttomLeftNewV, buttomRightNewV)
            buttom = min(hNew + 2 * hTurn, buttom)
            left = min(topLeftU, topRightU, buttomLeftU, buttomRightU, topLeftNewU, topRightNewU, buttomLeftNewU, buttomRightNewU)
            left = max(0, left)
            right = max(topLeftU, topRightU, buttomLeftU, buttomRightU, topLeftNewU, topRightNewU, buttomLeftNewU, buttomRightNewU)
            right = min(wNew + 2 * wTurn, right)

            self.topList.append(int(top))
            self.buttomList.append(int(buttom))
            self.leftList.append(int(left))
            self.rightList.append(int(right))

        else:
            print('count:', self.count, 'use existing data')
            H = self.HList[self.count]
            background[hTurn:hTurn + hTurn, wTurn:wTurn + wTurn, :] = imgTurn
            background = cv2.warpPerspective(background, H, (background.shape[1], background.shape[0]))

        temp = np.zeros((hNew + 2 * hTurn, wNew + 2 * wTurn, 3), dtype='uint8') # 对 imgNew 也要进行扩张，计算的H是相对imgNew的形状，如果不扩张，扩张后的imgTurn，原点还是对应扩张前imgNew的原点
        temp[hTurn:hTurn + hNew, wTurn:wTurn + wNew, :] = imgNew
        # cv2.imshow('temp', cv2.resize(temp, (int(temp.shape[1] * 0.3), int(temp.shape[0] * 0.3))))
        background[temp>0] = temp[temp>0]
        # cv2.imshow('background1', cv2.resize(background, (int(background.shape[1] * 0.3), int(background.shape[0] * 0.3))))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.figure()
        # plt.imshow(background)
        # plt.show()
        # 4.cut img border
        background = background[self.topList[self.count]:self.buttomList[self.count], self.leftList[self.count]:self.rightList[self.count], :]
        # cv2.imshow('background2', cv2.resize(background, (int(background.shape[1] * 0.3), int(background.shape[0] * 0.3))))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.figure()
        # plt.imshow(background)
        # plt.show()

        if self.count < self.imgNum - 2:
            if self.FirstLoop:
                if self.count == 0:
                    self.centerU = int((2 * wTurn + wNew) / 2)
                    self.centerV = int((2 * hTurn + hNew) / 2)
                    self.centerU -= self.leftList[self.count]
                    self.centerV -= self.topList[self.count]
                else:
                    self.centerU += wTurn
                    self.centerV += hTurn
                    self.centerU -= self.leftList[self.count]
                    self.centerV -= self.topList[self.count]
                # radius = 10
                # color = (0, 255, 255)
                # thickness = 2
                # blank = cv2.circle(background, (self.centerU, self.centerV), radius, color, thickness)
                # cv2.imshow('blank', cv2.resize(blank, (int(blank.shape[1] * 0.3), int(blank.shape[0] * 0.3))))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            self.count += 1
        else:
            if self.FirstLoop:
                if self.count == 0:
                    self.centerU = int((2 * wTurn + wNew) / 2)
                    self.centerV = int((2 * hTurn + hNew) / 2)
                    self.centerU -= self.leftList[self.count]
                    self.centerV -= self.topList[self.count]
                else:
                    self.centerU += wTurn
                    self.centerV += hTurn
                    self.centerU -= self.leftList[self.count]
                    self.centerV -= self.topList[self.count]
                # radius = 10
                # color = (0, 255, 255)
                # thickness = 2
                # blank = cv2.circle(background, (self.centerU, self.centerV), radius, color, thickness)
                # cv2.imshow('blank', cv2.resize(blank, (int(blank.shape[1] * 0.3), int(blank.shape[0] * 0.3))))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            self.FirstLoop = False
            self.count = 0

        return background

    def fusion(self, imgNew, imgTurn):
        hNew, wNew = imgNew.shape[:2]
        hTurn, wTurn = imgTurn.shape[:2]
        # 1.prepare a big picture
        background = np.zeros((hNew + 2 * hTurn, wNew + 2 * wTurn, 3), dtype='uint8')
        # 2.turn img
        if len(self.HList) < self.count + 1:
            print('count:', self.count, 'generate data')
            # 模块一：提取特征
            kpsNew, featuresNew = detect(imgNew)
            kpsTurn, featuresTurn = detect(imgTurn)
            # cv2.imshow('img_left', img_left)
            # cv2.imshow('img_right', img_right)
            # cv2.waitKey(2000)

            # 模块二：特征匹配
            matches, H, good = match_keypoints(kpsNew,kpsTurn,featuresNew,featuresTurn,0.5,0.99)
            print('matches:', len(matches))
            # ============================
            # img = cv2.drawMatchesKnn(img_left,kps_left,img_right,kps_right,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.figure(figsize=(20,20))
            # plt.imshow(img)
            # ============================
            # 模块三：透视变换-拼接
            self.HList.append(H)
            # ============================
            # show
            # plt.figure(figsize= size)
            # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            # plt.show()
            # ============================
            # H = HList[self.count]
            turnLeft = wNew + 2 * wTurn #左边界
            turnRight = 0              #右边界
            turnBottom = 0            #底部
            turnTop = hNew + 2 * hTurn #顶部
            ts = time.time()
            for v in range(hTurn):
                for u in range(wTurn):
                    newV = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])
                    newU = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])
                    newV += hTurn
                    newU += wTurn
                    if newU < turnLeft:
                        turnLeft = newU
                    if newU > turnRight:
                        turnRight = newU
                    if newV > turnBottom:
                        turnBottom = newV
                    if newV < turnTop:
                        turnTop = newV
                    newV = int(newV)
                    newU = int(newU)
                    if newV >= hNew + 2 * hTurn or newU >= wNew + 2 * wTurn:
                        continue
                    background[newV, newU, :] = imgTurn[v, u, :]
            print('time:', time.time() - ts)
            left = min(turnLeft, wTurn)
            right = max(turnRight, wTurn + wNew)
            top = min(turnTop, hTurn)
            buttom = max(turnBottom, hTurn + hNew)
            self.topList.append(int(top))
            self.buttomList.append(int(buttom))
            self.leftList.append(int(left))
            self.rightList.append(int(right))
        else:
            print('count:', self.count, 'use existing data')
            H = self.HList[self.count]
            ts = time.time()
            for v in range(hTurn):
                for u in range(wTurn):
                    newV = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])
                    newU = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / (H[2, 0] * u + H[2, 1] * v + H[2, 2])
                    newV += hTurn
                    newU += wTurn
                    newV = int(newV)
                    newU = int(newU)
                    if newV >= hNew + 2 * hTurn or newU >= wNew + 2 * wTurn:
                        continue
                    background[newV, newU, :] = imgTurn[v, u, :]
            print('time:', time.time() - ts)
        # cv2.imshow('background0', cv2.resize(background, (int(background.shape[1] * 0.6), int(background.shape[0] * 0.6))))
        # cv2.waitKey(0)
        # 3.put new img
        background[hTurn:hTurn + hNew, wTurn:wTurn + wNew, :] = imgNew
        # plt.figure()
        # plt.imshow(background)
        # plt.show()
        # 4.cut img border
        background = background[self.topList[self.count]:self.buttomList[self.count], self.leftList[self.count]:self.rightList[self.count], :]
        # background	= cv2.medianBlur(background, 7)
        # plt.figure()
        # plt.imshow(background)
        # plt.show()

        if self.count < self.imgNum - 2:
            self.count += 1
        else:
            if self.centerU == -1:
                self.centerU = int((2 * wTurn + wNew) / 2 - self.leftList[self.count])
                self.centerV = int((2 * hTurn + hNew) / 2 - self.topList[self.count])
            self.count = 0
        return background

    def fusionMulImg(self, imgList, dealOrder):
        baseImg = imgList[dealOrder[0]]
        for i in range(len(imgList) - 1):
            # turnImg = self.fusion(imgList[i + 1], turnImg)
            baseImg = self.fusionByAffine(baseImg, imgList[dealOrder[i + 1]])
            # print('turnImg shape:', turnImg.shape)
            # cv2.imshow('turnImg', cv2.resize(turnImg, (int(turnImg.shape[1] * 0.6), int(turnImg.shape[0] * 0.6))))
            # cv2.waitKey(0)
        # cv2.imshow('turnImg1', cv2.resize(turnImg, (int(turnImg.shape[1] * 0.6), int(turnImg.shape[0] * 0.6))))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        baseImg = self.projectToCylinder(baseImg, 79/180 * math.pi, self.centerU, self.centerV)
        # cv2.imshow('turnImg2', cv2.resize(turnImg, (int(turnImg.shape[1] * 0.6), int(turnImg.shape[0] * 0.6))))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return baseImg

def drawMatches(img_left, img_right, H):
    # 获取图片宽度和高度
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    """对imgB进行透视变换
    由于透视变换会改变图片场景的大小，导致部分图片内容看不到
    所以对图片进行扩展:高度取最高的，宽度为两者相加"""
    image = np.zeros((max(h_left, h_right), w_left+w_right, 3), dtype='uint8')
    # 初始化
    image[0:h_right, 0:w_right] = img_right
    """利用以获得的单应性矩阵进行变透视换"""
    image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))#(w,h
    """将透视变换后的图片与另一张图片进行拼接"""
    image[0:h_left, 0:w_left] = img_left
    # image = remove_black_edges(image)
    return image

def fusionImg(img_left, img_right, size=(20,20)):
    # 模块一：提取特征
    kps_left, features_left = detect(img_left)
    kps_right, features_right = detect(img_right)
    # cv2.imshow('img_left', img_left)
    # cv2.imshow('img_right', img_right)
    # cv2.waitKey(2000)

    # 模块二：特征匹配
    matches, H, good = match_keypoints(kps_left,kps_right,features_left,features_right,0.5,0.99)
    print('matches:', len(matches))
    img = cv2.drawMatchesKnn(img_left,kps_left,img_right,kps_right,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.figure(figsize=(20,20))
    # plt.imshow(img)
    # 模块三：透视变换-拼接
    vis = drawMatches(img_left, img_right, H)
    # show
    # plt.figure(figsize= size)
    # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    # plt.show()
    return vis, H

def fusionImgWithH(img_left, img_right, H):
    # 模块三：透视变换-拼接
    vis = drawMatches(img_left, img_right, H)
    # show
    # plt.figure(figsize= size)
    # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    # plt.show()
    return vis

# def fusionMulImg(imgList, HList = []):
#     if len(HList) == 0:
#         leftImg = imgList[0]
#         leftImg, H = fusionImg(leftImg, imgList[1])
#         for i in range(len(imgList) - 1):
#             leftImg, H = fusionImg(leftImg, imgList[i + 1])
#             HList.append(H)
#     else:
#         leftImg = imgList[0]
#         for i in range(len(imgList) - 1):
#             leftImg = fusionImgWithH(leftImg, imgList[i + 1], HList[i])

#     return leftImg, HList
def fusionMulImg(imgList, HList = []):
    if len(HList) == 0:
        rightImg = imgList[0]
        for i in range(len(imgList) - 1):
            rightImg, H = fusionImg(imgList[i + 1], rightImg)
            HList.append(H)
    else:
        rightImg = imgList[0]
        for i in range(len(imgList) - 1):
            rightImg = fusionImgWithH(imgList[i + 1], rightImg, HList[i])

    return rightImg, HList


# class Fusioner():
#     def __init__(self, imgList):
#         self.H = []
#         for i in range(len(imgList) - 1):
            
if __name__ == "__main__":
    suffix = '2021031501Test'
    # videoList = ['D:/project/videoFusion/data/2020122914/video/un20201229140.mp4','D:/project/videoFusion/data/2020122912/video/un20201229120.mp4','D:/project/videoFusion/data/2020122911/video/un20201229110.mp4']
    # dealOrder = [1,0,2]
    videoList = ['D:/project/videoFusion/data/2021031501/video/unE61605546-132-0.mp4','D:/project/videoFusion/data/2021031501/video/unE61605498-101-2.mp4','D:/project/videoFusion/data/2021031501/video/unE61605517-41-1.mp4']
    dealOrder = [2,1,0]

    readers = []
    for video in videoList:
        readers.append(readInput.InputReader(video))

    videoSavePath = '../output'
    imgFusioner = ImgFusioner(len(videoList))
    frameID = 0
    while frameID < 100:
        imgList = []
        for reader in readers:
            frame, bStop = reader.read()
            frame = cv2.resize(frame, (int(frame.shape[1]*0.6), int(frame.shape[0]*0.6)))
            imgList.append(frame)
        img = imgFusioner.fusionMulImg(imgList, dealOrder)
        if frameID == 0:
            imgSize = (img.shape[1], img.shape[0])
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoOut = cv2.VideoWriter()
            fps = 25
            videoOut.open(videoSavePath + '/' + suffix + '.mp4', fourcc, fps, imgSize, True)
            videoOut.write(img)
        else:
            videoOut.write(img)
        frameID += 1
        print('frameID:', frameID, 'end, time:', datetime.datetime.now())
    videoOut.release()

# def myStitch(imgList):
#     stitcher = cv2.Stitcher.create(0)
#     retval, pano = stitcher.stitch(imgList)
#     cv2.imshow('pano', pano)
#     cv2.waitKey(0)
#     return pano


# if __name__ == "__main__":
#     suffix = 'un_indoor_4_opencv'
#     # right first
#     # videoList = ['../data/videos5/5.mp4', '../data/videos5/4.mp4', '../data/videos5/3.mp4', '../data/videos5/2.mp4', '../data/videos5/1.mp4', '../data/videos5/0.mp4']
#     # videoList = ['../data/videos5/5.mp4', '../data/videos5/4.mp4', '../data/videos5/3.mp4']
#     # videoList = ['../data/20210107mul01/video/20210107mul011.mp4', '../data/20210107mul01/video/20210107mul010.mp4']'D:/project/videoFusion/data/2020122911/video/un20201229110.mp4', 
#     # videoList = ['D:/project/videoFusion/data/2020122902/video/un20201229020.mp4','D:/project/videoFusion/data/2020122901/video/un20201229010.mp4']
#     videoList = ['D:/project/videoFusion/data/2020122914/video/un20201229140.mp4','D:/project/videoFusion/data/2020122912/video/un20201229120.mp4']
#     # videoList = ['D:/project/videoFusion/data/2021012105/video/un20210121050.mp4',
#     # 'D:/project/videoFusion/data/2021012104/video/un20210121040.mp4',
#     # 'D:/project/videoFusion/data/2021012103/video/un20210121030.mp4',
#     # 'D:/project/videoFusion/data/20210121025/video/un202101210250.mp4']
#     # 'D:/project/videoFusion/data/2021012102/video/un20210121020.mp4',
#     # 'D:/project/videoFusion/data/2021012101/video/un20210121010.mp4',
#     # 'D:/project/videoFusion/data/2021012100/video/un20210121000.mp4'
#     # ]
#     readers = []
#     for video in videoList:
#         readers.append(readInput.InputReader(video))
#     # v1 = readInput.InputReader('../data/videos5/testRight.mp4')
#     # v2 = readInput.InputReader('../data/videos5/testCenter.mp4')
#     # v3 = readInput.InputReader('../data/videos5/testLeft.mp4')

#     videoSavePath = '../output'
#     HList = []
#     frameID = 0
#     while frameID < 100:
#         imgList = []
#         for reader in readers:
#             frame, bStop = reader.read()
#             frame = cv2.resize(frame, (int(frame.shape[1]*0.6), int(frame.shape[0]*0.6)))
#             imgList.append(frame)
#         # frame1, bStop = v1.read()
#         # frame1 = cv2.resize(frame1, (int(frame1.shape[1]*0.6), int(frame1.shape[0]*0.6)))
#         # frame2, bStop = v2.read()
#         # frame2 = cv2.resize(frame2, (int(frame2.shape[1]*0.6), int(frame2.shape[0]*0.6)))
#         # frame3, bStop = v3.read()
#         # frame3 = cv2.resize(frame3, (int(frame3.shape[1]*0.6), int(frame3.shape[0]*0.6)))
#         # show_points(frame1)
#         # show_points(frame2)

#         # imgList.append(frame1)
#         # imgList.append(frame2)
#         # imgList.append(frame3)
#         # img, HList= fusionMulImg(imgList, HList)
#         img = myStitch(imgList)
#         if frameID == 0:
#             imgSize = (img.shape[1], img.shape[0])
#             fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#             videoOut = cv2.VideoWriter()
#             fps = 25
#             videoOut.open(videoSavePath + '/' + suffix + '.mp4', fourcc, fps, imgSize, True)
#             videoOut.write(img)
#         else:
#             videoOut.write(img)
#         frameID += 1
#         print('frameID:', frameID, 'end, time:', datetime.datetime.now())
#     videoOut.release()