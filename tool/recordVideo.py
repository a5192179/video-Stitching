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
    else:
        shutil.rmtree(saveFolder)
        os.mkdir(saveFolder)

if __name__ == "__main__":
    # 0/init
    # inputStreams = ['rtsp://admin:1234abcd@192.168.1.132:554/Streaming/Channels/1',
    #                 'rtsp://admin:1234abcd@192.168.1.41:554/Streaming/Channels/1'
    #                 ]
    # inputStreams = ['rtsp://admin:1234abcd@192.168.1.101:554/Streaming/Channels/1',
    #                 'rtsp://admin:1234abcd@192.168.1.102:554/Streaming/Channels/1'
    #                 ]
    # inputStreams = ['0',
    #                 ]
    inputStreams = ['rtsp://admin:1234abcd@192.168.1.132:554/Streaming/Channels/1']
    sub = '20210121025'
    # inputStream = '0'
    saveFolder = '../data/' + sub
    initFolder(saveFolder)
    imgSavePath = saveFolder + '/img'
    initFolder(imgSavePath)
    videoSavePath = saveFolder + '/video'
    initFolder(videoSavePath)

    readers = []
    videoID = 0
    videoOuts = []
    for inputStream in inputStreams:
        inputReader = readInput.InputReader(inputStream)
        fps = inputReader.getFPS()
        # countFrame = -1
        width, high = inputReader.getSize()
        widthSave = int(width * 1.0)
        highSave = int(high * 1.0)
        imgSize = (widthSave, highSave)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoOut = cv2.VideoWriter()
        videoOut.open(videoSavePath + '/' + sub + str(videoID) + '.mp4', fourcc, fps, imgSize, True)
        readers.append(inputReader)
        videoOuts.append(videoOut)
        videoID += 1

    frameID = 0
    while frameID < 500:
        videoID = 0
        for inputReader in readers:
            # 1/read camera every second
            videoOut = videoOuts[videoID]
            frameOri, bStop = inputReader.read()
            if bStop:
                break
            frame = frameOri[int((high - highSave) / 2):high - int((high - highSave) / 2),int((width - widthSave) / 2):width - int((width - widthSave) / 2),:]
            cv2.imwrite(imgSavePath + '/' + str(videoID) + '-' + str(frameID) + '.jpg', frame)
            videoOut.write(frame)
            videoID += 1
        frameID += 1
    
        # print('video time:', time.time() - ts)
        # print('save time:', time.time() - ts_save)
        print('--------frameID:', frameID, ' end------------')

    videoOut.release()    