import cv2
import sys
sys.path.append('.')
import os
import shutil
from common import readInput
import datetime
import time

def initFolder(saveFolder):
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    # else:
    #     shutil.rmtree(saveFolder)
    #     os.mkdir(saveFolder)

if __name__ == "__main__":
    # 0/init
    # inputStream = 'rtsp://admin:1234abcd@192.168.1.132:554/Streaming/Channels/1'
    # inputStream = '0'
    inputStream = 'rtsp://admin:1234abcd@192.168.1.41:554/Streaming/Channels/1'
    # inputStream = 'rtsp://admin:1234abcd@192.168.1.132:554/Streaming/Channels/1'
    sub = 'E61605517-41'
    # inputStream = '0'
    saveFolder = '../data/Photo'
    initFolder(saveFolder)
    imgSavePath = saveFolder + '/cali/' + sub
    initFolder(imgSavePath)
    # videoSavePath = saveFolder + '/video'
    # initFolder(videoSavePath)

    inputReader = readInput.InputReader(inputStream)
    nextTime = time.time()
    fps = inputReader.getFPS()
    # countFrame = -1
    width, high = inputReader.getSize()
    widthSave = int(width * 1.0)
    highSave = int(high * 1.0)
    # imgSize = (widthSave, highSave)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # videoOut = cv2.VideoWriter()
    # videoID = 0
    # videoOut.open(videoSavePath + '/' + sub + str(videoID) + '.mp4', fourcc, fps, imgSize, True)
    timeSufix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    frameOri, bStop = inputReader.read()
    frame = frameOri[int((high - highSave) / 2):high - int((high - highSave) / 2),int((width - widthSave) / 2):width - int((width - widthSave) / 2),:]
    imgFile = imgSavePath + '/' + timeSufix + '.jpg'
    cv2.imwrite(imgFile, frame)
    print(imgFile, 'done')
    # while frameID < 500:
    #     # 1/read camera every second
    #     frameOri, bStop = inputReader.read()
    #     if bStop:
    #         break
    #     frame = frameOri[int((high - highSave) / 2):high - int((high - highSave) / 2),int((width - widthSave) / 2):width - int((width - widthSave) / 2),:]
    #     cv2.imwrite(imgSavePath + '/' + str(frameID) + '.jpg', frame)
    #     frameID += 1
    #     videoOut.write(frame)
    
    #     # print('video time:', time.time() - ts)
    #     # print('save time:', time.time() - ts_save)
    #     print('--------frameID:', frameID, ' end------------')

    # videoOut.release()    