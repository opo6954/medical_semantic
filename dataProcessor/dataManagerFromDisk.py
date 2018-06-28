# Data manager from disk file
# from label file
import random
import cv2
import numpy as np

class DataManagerFromDisk:
    def __init__(self, labelFilePath, batchSize=16, totalClass=6):
        self.labelFilePath = labelFilePath
        self.batchSize = batchSize
        self.totalClass = totalClass
        self.curIdx = 0
        self.imgNLabelList = []

        self.loadLabel()

    def getLengthOfAllImg(self):
        return len(self.imgNLabelList)

    def loadLabel(self):


        with open(self.labelFilePath) as readFile:
            allLabel = readFile.readlines()

        for eachLabel in allLabel:
            eachLabelList = eachLabel.split(',')

            imgPath = eachLabelList[0]
            label = eachLabelList[1]

            # one hot encoding...
            one_hot = []

            for _ in range(self.totalClass):
                one_hot.append(0.0)

            one_hot[int(label)] = 1.0

            # push img path and one-hot encoding to list
            self.imgNLabelList.append((imgPath, one_hot))

        # shuffle img and label list
        random.shuffle(self.imgNLabelList)

    def returnNextBatch(self):
        batchImgList = []
        batchLabelList = []

        for k in range(self.batchSize):
            if(self.curIdx >= self.getLengthOfAllImg()):
                self.curIdx = 0
                random.shuffle(self.imgNLabelList)

            currPair = self.imgNLabelList[self.curIdx]
            imgPath, label = (currPair[0], currPair[1])

            imgFile = cv2.imread(imgPath)

            if(imgFile is None):
                print('Not found image file for ' + imgPath)
                continue

            # batchList.append((imgFile, label))
            batchImgList.append(imgFile)
            batchLabelList.append(label)
            self.curIdx = self.curIdx + 1

        batchImgListNpy = np.asarray(batchImgList)
        batchLabelListNpy = np.asarray(batchLabelList)

        return (batchImgListNpy, batchLabelListNpy)








