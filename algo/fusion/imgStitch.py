import cv2
def myStitch(imgList, bSave, bShow, savePath = '.'):
    # PANORAMA = 0,
    # SCANS = 1
    stitcher = cv2.Stitcher.create(0)
    retval0, pano0 = stitcher.stitch(imgList)
    # cv2.imshow('pano', pano)
    # cv2.waitKey(0)

    stitcher = cv2.Stitcher.create(1)
    retval1, pano1 = stitcher.stitch(imgList)
    if bShow and retval0 + retval1 == 0:
        if retval0 == 0:
            cv2.imshow('pano0', pano0)
        if retval1 == 0:
            cv2.imshow('pano1', pano1)
        cv2.waitKey(0)
    if bSave:
        savePath0 = savePath + '/0.jpg'
        savePath1 = savePath + '/1.jpg'
        if retval0 == 0:
            cv2.imwrite(savePath0, pano0)
            print('0 success')
        else:
            print('0 fail')
        if retval1 == 0:
            cv2.imwrite(savePath1, pano1)
            print('1 success')
        else:
            print('1 fail')
    return pano0

if __name__ == "__main__":
    # imgPathList = ['D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/newspaper1.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/newspaper2.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/newspaper3.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/newspaper4.jpg']
    # imgPathList = ['D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/boat1.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/boat2.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/boat3.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/boat4.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/boat5.jpg',
    # 'D:/project/videoFusion/data/opencv_extra-master/testdata/stitching/boat6.jpg']

    imgPathList = ['D:/project/videoFusion/data/corner/2.jpg',
    'D:/project/videoFusion/data/corner/4.jpg']
    bSave = True
    savePath = '../output/boat'
    bShow = False
    imgList = []
    for imgPath in imgPathList:
        img = cv2.imread(imgPath)
        imgList.append(img)

    img = myStitch(imgList, bSave, bShow, savePath)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)