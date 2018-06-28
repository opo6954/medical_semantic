'''
make data input with activated scheme with bbox information
inputPath: inputPath(with imgFololder and roiFolder exist)
outputPath: outptuPath
'''

import numpy as np
import cv2
import os

imgFolderName = 'PNG'
roiFolderName = 'TXT'
maskFolderName = 'MASK'


def makeCroppedSchemeWithMask(inputPath, outputPath):
    maskFolderAbsPath = os.path.join(inputPath, maskFolderName)
    maskFolderList = os.listdir(maskFolderAbsPath)

    maskFolderList.sort()

    maskList = []

    print('load for mask file with ' + inputPath)


    prevMaskFileName = ''
    orderIdx=0

    for maskFileName in maskFolderList:
        maskFileAbsPath = os.path.join(maskFolderAbsPath, maskFileName)

        print('load for mask file of ' + maskFileAbsPath)

        imgMask = cv2.imread(maskFileAbsPath)

        if (imgMask is None):
            continue

        tmp = maskFileName.split('.')[0]
        tmpList = tmp.split('_')

        fileName = tmpList[0] + '_' + tmpList[1]

        if (prevMaskFileName == fileName):
            orderIdx = orderIdx + 1
        else:
            prevMaskFileName = fileName
            orderIdx = 0

        roiIdx = int(tmpList[2])

        maskList.append((fileName, roiIdx, imgMask, orderIdx))

    for maskTuple in maskList:
        fileName = maskTuple[0]
        roiIdx = maskTuple[1]
        imgMask = maskTuple[2]
        orderIdx = maskTuple[3]

        imgAbsPath = os.path.join(inputPath, imgFolderName, fileName + '.png')

        print('On ' + fileName + "_" + str(roiIdx))

        if (not os.path.exists(imgAbsPath)):
            print('Not exist for ' + imgAbsPath)
            continue

        img = cv2.imread(imgAbsPath)

        if (not img.shape == imgMask.shape):
            imgMask = cv2.resize(imgMask, (img.shape[1], img.shape[0]))

        mask_res = cv2.bitwise_or(img, imgMask)

        # load bbox information

        roiAbsPath = os.path.join(inputPath, roiFolderName, fileName + '.txt')
        with open(roiAbsPath) as roiFile:
            roiFileAllContents = roiFile.readlines()
            bboxIdx = int(roiFileAllContents[0])

            if(orderIdx < bboxIdx):
                x1, y1, x2, y2 = roiFileAllContents[int(orderIdx)+1].split()
            else:
                x1, x2, y1, y2 = (0, 0, 0, 0)
                print('Not exist roi file with ' + roiAbsPath)
        mask_res_cropped = mask_res[int(y1):int(y2), int(x1):int(x2)]


        outputActivateAbsPath = os.path.join(outputPath, fileName + "_" + str(roiIdx) + '.png')

        cv2.imwrite(outputActivateAbsPath, mask_res_cropped)

def makeActivatedSchemeWithMask(inputPath, outputPath):
    maskFolderAbsPath = os.path.join(inputPath, maskFolderName)
    maskFolderList = os.listdir(maskFolderAbsPath)

    maskFolderList.sort()

    maskList = []

    # load mask file

    print('load for mask file...')

    prevMaskFileName = ''
    orderIdx=0

    for maskFileName in maskFolderList:
        maskFileAbsPath = os.path.join(maskFolderAbsPath, maskFileName)

        print('load for mask file of ' + maskFileAbsPath)

        imgMask = cv2.imread(maskFileAbsPath)

        if(imgMask is None):
            continue


        tmp = maskFileName.split('.')[0]
        tmpList = tmp.split('_')
        fileName = tmpList[0] + '_' + tmpList[1]

        if (prevMaskFileName == fileName):
            orderIdx = orderIdx + 1
        else:
            prevMaskFileName = fileName
            orderIdx = 0

        roiIdx = int(tmpList[2])

        maskList.append((fileName, roiIdx, imgMask, orderIdx))


    for maskTuple in maskList:
        fileName = maskTuple[0]
        roiIdx = maskTuple[1]
        imgMask = maskTuple[2]

        imgAbsPath = os.path.join(inputPath, imgFolderName, fileName + '.png')

        print('On ' + fileName + "_" + str(roiIdx))

        if(not os.path.exists(imgAbsPath)):
            print('Not exist for ' + imgAbsPath)
            continue

        img = cv2.imread(imgAbsPath)

        if(not img.shape == imgMask.shape):
            imgMask = cv2.resize(imgMask, (img.shape[1], img.shape[0]))


        mask_res = cv2.bitwise_or(img, imgMask)


        outputActivateAbsPath = os.path.join(outputPath, fileName + "_" + str(roiIdx) + '.png')



        cv2.imwrite(outputActivateAbsPath, mask_res)




        # load origin image








#         load mask image

# make activated scheme into one single outputPath together
def makeActivatedSchemeWithBbox(inputPath, outputPath):
    # load ROI folder
    roiFolderAbsPath = os.path.join(inputPath, roiFolderName)
    roiFolderList = os.listdir(roiFolderAbsPath)

    roiFolderList.sort()

    bboxList = []

    # load roi file(bbox file)


    for roiFileName in roiFolderList:
        roiFileAbsPath = os.path.join(roiFolderAbsPath, roiFileName)


        # load roi
        # roi format:
        # number of roi
        # roi bounding box(x1, y1, x2, y2)


        # load roi and save it to list
        with open(roiFileAbsPath) as roiFile:
            allContents = roiFile.readlines()
            roiCount = int(allContents[0])

            for k in range(roiCount):
                (x1, y1, x2, y2) = allContents[k+1].split()
                # only take the roi file name(excluding extension)
                bboxList.append((roiFileName.split('.')[0], k, int(x1), int(y1), int(x2), int(y2)))


    # load image file
    imgFolderAbsPath = os.path.join(inputPath, imgFolderName)

    for bbox in bboxList:
        roiFileName = bbox[0]
        bboxIdx = bbox[1]
        (x1, y1, x2, y2) = (bbox[2], bbox[3], bbox[4], bbox[5])



        imgFileAbsPath = os.path.join(imgFolderAbsPath, roiFileName + '.png')

        if(not os.path.exists(imgFileAbsPath)):
            print('Not exist ' + imgFileAbsPath)
            continue

        img = cv2.imread(imgFileAbsPath)

        # last of region, set zero value...
        imgActivated = np.zeros(img.shape, np.uint8)

        # activate image with bbox information
        imgActivated[y1:y2, x1:x2] = img[y1:y2, x1:x2]

        # debug for imgActivated
        # img = cv2.resize(img, (224,224))
        # imgActivated = cv2.resize(imgActivated, (224, 224))
        # cv2.imshow("origin", img)
        # cv2.imshow("activated", imgActivated)
        # cv2.waitKey()


        outputActivateAbsPath = os.path.join(outputPath, roiFileName + "_" + str(bboxIdx) + '.png')

        print('save for ' + roiFileName + "_" + str(bboxIdx))
        cv2.imwrite(outputActivateAbsPath, imgActivated)

def makeCroppedSchemeWithBbox(inputPath, outputPath):
    # load ROI folder
    roiFolderAbsPath = os.path.join(inputPath, roiFolderName)
    roiFolderList = os.listdir(roiFolderAbsPath)

    roiFolderList.sort()

    bboxList = []

    # load roi file(bbox file)


    for roiFileName in roiFolderList:
        roiFileAbsPath = os.path.join(roiFolderAbsPath, roiFileName)

        # load roi
        # roi format:
        # number of roi
        # roi bounding box(x1, y1, x2, y2)


        # load roi and save it to list
        with open(roiFileAbsPath) as roiFile:
            allContents = roiFile.readlines()
            roiCount = int(allContents[0])

            for k in range(roiCount):
                (x1, y1, x2, y2) = allContents[k + 1].split()
                # only take the roi file name(excluding extension)
                bboxList.append((roiFileName.split('.')[0], k, int(x1), int(y1), int(x2), int(y2)))

    # load image filefor bbox in bboxList:

    imgFolderAbsPath = os.path.join(inputPath, imgFolderName)

    for bbox in bboxList:
        roiFileName = bbox[0]
        bboxIdx = bbox[1]
        (x1, y1, x2, y2) = (bbox[2], bbox[3], bbox[4], bbox[5])

        imgFileAbsPath = os.path.join(imgFolderAbsPath, roiFileName + '.png')

        if (not os.path.exists(imgFileAbsPath)):
            print('Not exist ' + imgFileAbsPath)
            continue

        img = cv2.imread(imgFileAbsPath)

        # last of region, set zero value...




        # cropped image with bbox information
        imgCropped = img[y1:y2, x1:x2]

        # debug for imgActivated
        # img = cv2.resize(img, (224,224))
        # imgActivated = cv2.resize(imgActivated, (224, 224))
        # cv2.imshow("origin", img)
        # cv2.imshow("activated", imgActivated)
        # cv2.waitKey()


        outputActivateAbsPath = os.path.join(outputPath, roiFileName + "_" + str(bboxIdx) + '.png')

        print('save for ' + roiFileName + "_" + str(bboxIdx))
        cv2.imwrite(outputActivateAbsPath, imgCropped)

# For generate cropped scheme with mask



makeCroppedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/1.nodule', outputPath='../wholeData_mask/cropped')
makeCroppedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/3.consolidation', outputPath='../wholeData_mask/cropped')
makeCroppedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/4.interstitial_opacity', outputPath='../wholeData_mask/cropped')
makeCroppedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/9.cardiomegaly', outputPath='../wholeData_mask/cropped')
makeCroppedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/10.pleural_effusion', outputPath='../wholeData_mask/cropped')
makeCroppedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/11.pneumothorax', outputPath='../wholeData_mask/cropped')




# For generate activated scheme with mask


makeActivatedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/1.nodule', outputPath='../wholeData_mask/activated')
makeActivatedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/3.consolidation', outputPath='../wholeData_mask/activated')
makeActivatedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/4.interstitial_opacity', outputPath='../wholeData_mask/activated')
makeActivatedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/9.cardiomegaly', outputPath='../wholeData_mask/activated')
makeActivatedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/10.pleural_effusion', outputPath='../wholeData_mask/activated')
makeActivatedSchemeWithMask(inputPath='../../../data_0326/ChestPA_6Class/11.pneumothorax', outputPath='../wholeData_mask/activated')




'''

# For generate cropped scheme with bbox
makeCroppedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/1.nodule', outputPath='../wholeData_bbox/cropped')
makeCroppedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/3.consolidation', outputPath='../wholeData_bbox/cropped')
makeCroppedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/4.interstitial_opacity', outputPath='../wholeData_bbox/cropped')
makeCroppedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/9.cardiomegaly', outputPath='../wholeData_bbox/cropped')
makeCroppedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/10.pleural_effusion', outputPath='../wholeData_bbox/cropped')
makeCroppedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/11.pneumothorax', outputPath='../wholeData_bbox/cropped')

# For generate activated scheme with bbox

makeActivatedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/1.nodule', outputPath='../wholeData_bbox/activated')
makeActivatedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/3.consolidation', outputPath='../wholeData_bbox/activated')
makeActivatedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/4.interstitial_opacity', outputPath='../wholeData_bbox/activated')
makeActivatedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/9.cardiomegaly', outputPath='../wholeData_bbox/activated')
makeActivatedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/10.pleural_effusion', outputPath='../wholeData_bbox/activated')
makeActivatedSchemeWithBbox(inputPath='../../data_asan_0326/ChestPA_6Class_0326/11.pneumothorax', outputPath='../wholeData_bbox/activated')

'''































