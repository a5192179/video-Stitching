import sys
sys.path.append('.')
from algo.fusion.imgStitch import myStitch
from common.file import initFolder
from common.readInput import InputReader
import cv2


# imgFolder = 'D:/project/videoFusion/data/Photo/wide3/undisorted'
imgFolder = 'D:/project/videoFusion/data/Photo/wide2/undisorted'
saveFolder = imgFolder + '/fusion'
initFolder(saveFolder)

inputReader = InputReader(imgFolder)
bStop = False
imgs = []
imgNumMax = 5
imgNum = 0
while not bStop:
    frame, bStop = inputReader.read()
    if bStop or imgNum == imgNumMax:
        break
    frame = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
    imgs.append(frame)
    imgNum += 1
print('--------imgNum:', imgNum, ' end------------')

# savePath = saveFolder + '/fusion.jpg'
bSave = True
bShow = False
myStitch(imgs, bSave, bShow, savePath = saveFolder)