import numpy as np
import cv2
import os
import tensorflow as tf

'''
from image and label file to npy file
'''


IMAGENET_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)


def convLabelFile2Npy(inputLabelFilePath, outputNpyFileParentPath, postfixName, totalLabel, isResize=True, resizeFactor=227):
    with open(inputLabelFilePath) as readFile:
        allFileNLabels = readFile.readlines()

    imgList = []
    labelList = []




    for singleFileNLabels in allFileNLabels:
        print('for load ' + singleFileNLabels)
        singleFileNLabelsList = singleFileNLabels.split(',')
        imgFilePath = singleFileNLabelsList[0]
        label = singleFileNLabelsList[1]

        img = cv2.imread(imgFilePath)

        # resize
        img_resized = cv2.resize(img, (resizeFactor, resizeFactor))

        img_resized = img_resized.astype(np.float32)

        # subtract with imagenet_mean
        img_centered = img_resized - IMAGENET_MEAN

        # change rgb to bgr
        img_bgr = img_centered[:,:,::-1]

        imgList.append(img_bgr)

        one_hot = []

        for _ in range(totalLabel):
            one_hot.append(0.0)

        one_hot[int(label)] = 1.0

        labelList.append(one_hot)


    imgListNpy = np.asarray(imgList)
    # imgListNpy = np.asarray(np.multiply(imgList, 1.0 / 255.0), dtype=np.float32)
    labelListNpy = np.asarray(labelList)

    imgAbsPath = os.path.join(outputNpyFileParentPath, postfixName + "_img.npy")
    labelAbsPath = os.path.join(outputNpyFileParentPath, postfixName + "_label.npy")

    print('save npy file...')
    np.save(file=imgAbsPath, arr=imgListNpy)
    np.save(file=labelAbsPath, arr=labelListNpy)

    print('save npy file done')


# convLabelFile2Npy(inputLabelFilePath='../dataLabelFiles/wholeData_bbox_cropped_train.label', outputNpyFileParentPath='../dataNpyFiles', postfixName='bbox_cropped_train', totalLabel=6)
# convLabelFile2Npy(inputLabelFilePath='../dataLabelFiles/wholeData_bbox_cropped_test.label', outputNpyFileParentPath='../dataNpyFiles', postfixName='bbox_cropped_test', totalLabel=6)

# convLabelFile2Npy(inputLabelFilePath='../dataLabelFiles/wholeData_bbox_activated_train.label', outputNpyFileParentPath='../dataNpyFiles', postfixName='bbox_activated_train', totalLabel=6)
# convLabelFile2Npy(inputLabelFilePath='../dataLabelFiles/wholeData_bbox_activated_test.label', outputNpyFileParentPath='../dataNpyFiles', postfixName='bbox_activated_test', totalLabel=6)

# convLabelFile2Npy(inputLabelFilePath='../dataLabelFiles/wholeData_origin_train.label', outputNpyFileParentPath='../dataNpyFiles', postfixName='origin_train', totalLabel=6)
# convLabelFile2Npy(inputLabelFilePath='../dataLabelFiles/wholeData_origin_test.label', outputNpyFileParentPath='../dataNpyFiles', postfixName='origin_test', totalLabel=6)




# convLabelFile2Npy(inputLabelFilePath='../exp_jh/bbox_query.label', outputNpyFileParentPath='../exp_jh', postfixName='bbox_query', totalLabel=6)
# convLabelFile2Npy(inputLabelFilePath='../exp_jh/bbox_search.label', outputNpyFileParentPath='../exp_jh', postfixName='bbox_search', totalLabel=6)

# convLabelFile2Npy(inputLabelFilePath='../exp_jh/activated_query.label', outputNpyFileParentPath='../exp_jh', postfixName='activated_query', totalLabel=6)
# convLabelFile2Npy(inputLabelFilePath='../exp_jh/activated_search.label', outputNpyFileParentPath='../exp_jh', postfixName='activated_search', totalLabel=6)

# convLabelFile2Npy(inputLabelFilePath='../exp_jh/origin_query.label', outputNpyFileParentPath='../exp_jh', postfixName='origin_query', totalLabel=6)
# convLabelFile2Npy(inputLabelFilePath='../exp_jh/origin_search.label', outputNpyFileParentPath='../exp_jh', postfixName='origin_search', totalLabel=6)

convLabelFile2Npy(inputLabelFilePath='../exp_cifar10/labelfile/cifar10_train.label', outputNpyFileParentPath='../exp_cifar10/npyfile', postfixName='cifar10_train', totalLabel=10)
convLabelFile2Npy(inputLabelFilePath='../exp_cifar10/labelfile/cifar10_test.label', outputNpyFileParentPath='../exp_cifar10/npyfile', postfixName='cifar10_test', totalLabel=10)


# q = np.load('../dataNpyFiles/bbox_cropped_train_img.npy')
# p = np.load('../dataNpyFiles/bbox_cropped_train_label.npy')

print('hello')









