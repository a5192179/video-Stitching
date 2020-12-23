import cv2
import sys
sys.path.append('.')
import os
import shutil
from module.detectFace.algo import detectFaceIF as detectFace
from module.detectSideFace.algo import detectSideFace
from common import readInput
import datetime
import time

def initFolder(saveFolder):
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    else:
        shutil.rmtree(saveFolder)
        os.mkdir(saveFolder)

if __name__ == "__main__":
    # 0/init
    inputStream = 'rtsp://admin:1234abcd@192.168.1.132:554/Streaming/Channels/1'
    # inputStream = '0'
    saveFolder = '../data/collectLS120701'
    initFolder(saveFolder)
    imgSavePath = saveFolder + '/img'
    initFolder(imgSavePath)
    faceSavePath = saveFolder + '/face'
    initFolder(faceSavePath)
    videoSavePath = saveFolder + '/video'
    initFolder(videoSavePath)

    faceDetector = detectFace.faceDetector()
    inputReader = readInput.InputReader(inputStream)
    sideFaceDetector = detectSideFace.sideFaceDetector()
    nextTime = time.time()
    delayOff = 2
    delay = delayOff
    fps = inputReader.getFPS()
    delayOn = 2 / fps
    delayNum = int(fps * 2)
    countFrame = -1
    width, high = inputReader.getSize()
    imgSize = (width, high)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoOut = cv2.VideoWriter()
    videoID = 0
    videoOut.open(videoSavePath + '/test' + str(videoID) + '.mp4', fourcc, fps, imgSize, True)
    faceID = 0
    frameID = 0
    faceSaveTime = time.time()
    faceSaveInterval = 0.25

    while True:
        # 1/read camera every second
        sleepTime = nextTime - time.time()
        print('sleepTime', sleepTime)
        if sleepTime > 0:
            time.sleep(sleepTime)
        readTime = time.time()
        frameOri, bStop = inputReader.read()
        if bStop:
            break
        frame = frameOri[50:800,150:1920-350,:]
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1000)
        # 2/detect face
        ts = time.time()
        oriFaces = faceDetector.detectFace(frame)
        faces = sideFaceDetector.getFrontFaces(oriFaces)
        print('detect time:', time.time() - ts)
        # 3/ change frequence to 25hz 
        if len(faces) > 0:
            delay = delayOn
            countFrame = 0
        elif countFrame >= 0 and countFrame < delayNum:
            delay = delayOn
            countFrame += 1
            print('face missing, countFrame:', countFrame)
        else:
            if countFrame == delayNum:
                countFrame = -1
            delay = delayOff
            nextTime = readTime + delay
            print('face miss, low power mode')
            continue
        nextTime = readTime + delay
        # 4/save face and video
        ts_save = time.time()
        if len(faces) > 0:
            print('face num:', len(faces))
            if readTime - faceSaveTime > faceSaveInterval:
                for face in faces:
                    cv2.imwrite(faceSavePath + '/' + str(faceID) + '.jpg', face)
                    faceID +=1
                faceSaveTime = readTime
            else:
                print('dont save face')
        cv2.imwrite(imgSavePath + '/' + str(frameID) + '.jpg', frame)
        frameID += 1
        ts = time.time()
        videoOut.write(frame)
        print('video time:', time.time() - ts)
        print('save time:', time.time() - ts_save)
        print('--------frameID:', frameID, ' end------------')
        if (frameID % ( 1 * 60 * fps) == 0):
            videoOut.release()
            videoOut = cv2.VideoWriter()
            videoOut.open(videoSavePath + '/test' + str(videoID) + '.mp4', fourcc, fps, imgSize, True)
            videoID += 1
            