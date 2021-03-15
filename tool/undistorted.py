import cv2
import sys
sys.path.append('.')
import os
import shutil
from common import readInput
import datetime
import time
import pickle

def initFolder(saveFolder):
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    else:
        shutil.rmtree(saveFolder)
        os.mkdir(saveFolder)

if __name__ == "__main__":
    # inputStreams = ['D:/project/videoFusion/data/2021031501/video/E61605546-132-0.mp4']
    # inputStreams = ['D:/project/videoFusion/data/2021031501/video/E61605498-101-2.mp4']
    inputStreams = ['D:/project/videoFusion/data/2021031501/video/E61605517-41-1.mp4']

    # caliImgFolder = '../data/Photo/cali/E61605546-132'
    # caliImgFolder = '../data/Photo/cali/E61605498-101'
    caliImgFolder = '../data/Photo/cali/E61605517-41'

    cameraID = caliImgFolder.split('/')[-1]
    loadFile = caliImgFolder + '/cameraMatrix' + cameraID + '.txt'
    with open(loadFile, 'rb') as f:
        matrix = pickle.load(f)
    loadFile = caliImgFolder + '/cameraDist' + cameraID + '.txt'
    with open(loadFile, 'rb') as f:
        dist = pickle.load(f)

    # 0/init
    # inputStreams = ['rtsp://admin:1234abcd@192.168.1.132:554/Streaming/Channels/1',
    #                 'rtsp://admin:1234abcd@192.168.1.41:554/Streaming/Channels/1'
    #                 ]
    # inputStreams = ['D:/project/videoFusion/data/2020122911/video/20201229110.mp4'
    #                 ]
    # inputStreams = ['D:/project/videoFusion/data/2020122902/video/20201229020.mp4'
    #                 ]
    # inputStreams = ['D:/project/videoFusion/data/20210121025/video/202101210250.mp4'
    #                 ]
    # inputStreams = ['D:/project/videoFusion/data/Photo/wide2']
    for inputStream in inputStreams:
        #img folder
        if os.path.isdir(inputStream):
            saveFolder = inputStream + '/undisorted'
            initFolder(saveFolder)

            inputReader = readInput.InputReader(inputStream)
            bStop = False
            frameID = 0
            while not bStop:
                frameOri, bStop = inputReader.read()
                imgPath = inputReader.getImgPath()
                imgName = imgPath.split('/')[-1]
                if bStop:
                    break
                frame = cv2.undistort(frameOri, matrix, dist)
                savePath = saveFolder + '/un' + imgName
                cv2.imwrite(savePath, frame)
                frameID += 1
                print('img:', savePath)
                print('--------frameID:', frameID, ' end------------')
            break
        #video
        temp = inputStream.split('/')
        folder = temp[0]
        for i in range(1, len(temp) - 1):
            folder = folder + '/' + temp[i]
        videoName = temp[-1]
        videoSavePath = folder + '/un' + videoName
        print('videoSavePath', videoSavePath)
        inputReader = readInput.InputReader(inputStream)
        fps = inputReader.getFPS()
        # countFrame = -1
        width, high = inputReader.getSize()
        imgSize = (width, high)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoOut = cv2.VideoWriter()
        videoOut.open(videoSavePath, fourcc, fps, imgSize, True)
        frameID = 0
        while frameID < 500:
            videoID = 0
            frameOri, bStop = inputReader.read()
            if bStop:
                break
            frame = cv2.undistort(frameOri, matrix, dist)
            videoOut.write(frame)
            frameID += 1
            print('--------frameID:', frameID, ' end------------')
        videoOut.release()
        