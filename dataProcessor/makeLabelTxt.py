import numpy as np
import os
import cv2
import random

'''
make label text file(txt including image path string, label(number)) with only image files

label number:
0: Nodule
1: Consolidation
2: InterstitialOpacity
3: Cardiomegaly
4: PleuralEffusion
5: Pneumothorax
'''

# Divide train and test set into 8:2

labelNameList = ['01Nodule', '03Consolidation', '04InterstitialOpacity', '09Cardiomegaly', '10PleuralEffusion', '11Pneumothorax']

trainRatio = 0.8



def makeLabelText(inputPath, outputPath):


    # load image path and assign label number according to labelNameList
    imgInputList = os.listdir(inputPath)

    random.shuffle(imgInputList)

    pathNLabelList = []

    count=0

    for imgInputSingle in imgInputList:
        print('for ' + imgInputSingle + ' is processing...')
        txtProcessing = imgInputSingle.split('_')[1]

        isExistLabel = False
        idx = -1

        for k in range(len(labelNameList)):
            if( labelNameList[k] in txtProcessing):
                isExistLabel = True
                idx = k
                absPath = os.path.join(inputPath, imgInputSingle)
                labelNumber = k
                pathNLabelList.append((absPath + ',' + str(labelNumber)))
                count=count+1
                break

        if(isExistLabel == False):
            print('Not exist of ' + txtProcessing)
            continue

    # write file with imgPath N label

    with open(outputPath, 'w') as writeFile:
        for eachPathNLabel in pathNLabelList:
            writeFile.write(eachPathNLabel + '\n')


def divideTrainNTest(inputPath, outputTrainPath, outputTestPath):
    # load image path and assign label number according to labelNameList
    imgInputList = os.listdir(inputPath)

    random.shuffle(imgInputList)

    pathNLabelTrainList = []
    pathNLabelTestList = []

    count = 0

    for imgInputSingle in imgInputList:
        print('for ' + imgInputSingle + ' is processing...')
        txtProcessing = imgInputSingle.split('_')[1].split('.')[0]
        # txtProcessing = imgInputSingle.split('_')[1]

        isExistLabel = False
        idx = -1

        for k in range(len(labelNameList)):
            if (labelNameList[k] in txtProcessing):
                isExistLabel = True
                idx = k
                absPath = os.path.join(inputPath, imgInputSingle)
                labelNumber = k
                if(count < len(imgInputList) * 0.8 ):
                    pathNLabelTrainList.append((absPath + ',' + str(labelNumber)))
                else:
                    pathNLabelTestList.append((absPath + ',' + str(labelNumber)))
                count = count + 1
                break

        if (isExistLabel == False):
            print('Not exist of ' + txtProcessing)
            continue

    # write file with imgPath N label

    with open(outputTestPath, 'w') as writeFile:
        for eachPathNLabel in pathNLabelTestList:
            writeFile.write(eachPathNLabel + '\n')

    with open(outputTrainPath, 'w') as writeFile:
        for eachPathNLabel in pathNLabelTrainList:
            writeFile.write(eachPathNLabel + '\n')


def divideTrainNTestTmp(inputPath, outputTrainPath, outputTestPath):
    # load image path and assign label number according to labelNameList
    imgInputList = os.listdir(inputPath)

    random.shuffle(imgInputList)

    pathNLabelTrainList = []
    pathNLabelTestList = []

    count = 0

    for imgInputSingle in imgInputList:
        print('for ' + imgInputSingle + ' is processing...')
        txtProcessing = imgInputSingle.split('_')[1]

        isExistLabel = False
        idx = -1

        for k in range(len(labelNameList)):
            if (labelNameList[k] in txtProcessing):
                isExistLabel = True
                idx = k
                absPath = os.path.join(inputPath, imgInputSingle)
                labelNumber = k
                if(count < len(imgInputList) * 0.8 ):
                    pathNLabelTrainList.append(('../' + absPath + ' ' + str(labelNumber)))
                else:
                    pathNLabelTestList.append(('../' + absPath + ' ' + str(labelNumber)))
                count = count + 1
                break

        if (isExistLabel == False):
            print('Not exist of ' + txtProcessing)
            continue

    # write file with imgPath N label

    with open(outputTestPath, 'w') as writeFile:
        for eachPathNLabel in pathNLabelTestList:
            writeFile.write(eachPathNLabel + '\n')

    with open(outputTrainPath, 'w') as writeFile:
        for eachPathNLabel in pathNLabelTrainList:
            writeFile.write(eachPathNLabel + '\n')

        def divideTrainNTest(inputPath, outputTrainPath, outputTestPath):
            # load image path and assign label number according to labelNameList
            imgInputList = os.listdir(inputPath)

            random.shuffle(imgInputList)

            pathNLabelTrainList = []
            pathNLabelTestList = []

            count = 0

            for imgInputSingle in imgInputList:
                print('for ' + imgInputSingle + ' is processing...')
                txtProcessing = imgInputSingle.split('_')[1]

                isExistLabel = False
                idx = -1

                for k in range(len(labelNameList)):
                    if (labelNameList[k] in txtProcessing):
                        isExistLabel = True
                        idx = k
                        absPath = os.path.join(inputPath, imgInputSingle)
                        labelNumber = k
                        if (count < len(imgInputList) * 0.8):
                            pathNLabelTrainList.append((absPath + ' ' + str(labelNumber)))
                        else:
                            pathNLabelTestList.append((absPath + ' ' + str(labelNumber)))
                        count = count + 1
                        break

                if (isExistLabel == False):
                    print('Not exist of ' + txtProcessing)
                    continue

            # write file with imgPath N label

            with open(outputTestPath, 'w') as writeFile:
                for eachPathNLabel in pathNLabelTestList:
                    writeFile.write(eachPathNLabel + '\n')

            with open(outputTrainPath, 'w') as writeFile:
                for eachPathNLabel in pathNLabelTrainList:
                    writeFile.write(eachPathNLabel + '\n')


# divideTrainNTest('../wholeData_bbox/activated', '../dataLabelFiles/wholeData_bbox_activated_train.label', '../dataLabelFiles/wholeData_bbox_activated_test.label')
# divideTrainNTestTmp(inputPath='../wholeData_bbox/cropped', outputTrainPath='../Alexnet/finetune_alexnet_with_tensorflow/train.txt', outputTestPath='../Alexnet/finetune_alexnet_with_tensorflow/test.txt')
# divideTrainNTest('../wholeData_mask/activated', '../dataLabelFiles/wholeData_mask_activated_train.label', '../dataLabelFiles/wholeData_mask_activated_test.label')
# divideTrainNTest('../wholeData_mask/cropped', '../dataLabelFiles/wholeData_mask_cropped_train.label', '../dataLabelFiles/wholeData_mask_cropped_test.label')
# divideTrainNTest('../wholeData_origin', '../dataLabelFiles/wholeData_origin_train.label', '../dataLabelFiles/wholeData_origin_test.label')





'''
# make label txt for activated, cropped of bbox, mask
makeLabelText('../wholeData_bbox/activated', '../dataLabelFiles/wholeData_bbox_activated.label')
makeLabelText('../wholeData_bbox/cropped', '../dataLabelFiles/wholeData_bbox_cropped.label')
makeLabelText('../wholeData_mask/activated', '../dataLabelFiles/wholeData_mask_activated.label')
makeLabelText('../wholeData_mask/cropped', '../dataLabelFiles/wholeData_mask_cropped.label')

# make label txt for origin file
makeLabelText('../wholeData_origin', '../dataLabelFiles/wholeData_origin.label')
'''








