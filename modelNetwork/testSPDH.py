import tensorflow as tf
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, '../dataProcessor')
from dataManager import DataManager



# test of SPDH

# function that input is labelPath, output is array of hash code and image name pair

def sigmoid2Bit(bitRepresentation):
    pass
def onehot2labelIdx(onehotLabel):
    pass

def filePath2Npy(filePath=''):
    IMAGENET_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)

    img = cv2.imread(filePath)

    img_resized = cv2.resize(img, (227,227))
    img_resized = img_resized.astype(np.float32)

    img_centered = img_resized - IMAGENET_MEAN
    img_bgr = img_centered[:,:,::-1]

    return img_bgr



# type of label file:
# ../wholeData_bbox/activated/021000069_01Nodule_0.png, 0
def extractSPDH(modelPath='', labelPath='', isTrain = True, additionalPrefix = '', outputParentPath=''):

    num_classes = 6
    batchSize = 32
    print('load model meta file...')

    # feed_forwarding
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(modelPath + '.meta')

    filePathList = []
    fileImgList = []
    labelList = []
    bitList = []

    print('load label file...')
    # load all file and label to list
    with open(labelPath) as labelFile:
        allLines = labelFile.readlines()

    for eachLine in allLines:
        eachLineSplit = eachLine.split(',')
        singleFilePath = eachLineSplit[0]
        singleLabel = eachLineSplit[1].rstrip()

        filePathList.append(singleFilePath)
        labelList.append(singleLabel)

    # convert npy image npy file
    print('load image file...')
    for eachFilePath in filePathList:
        eachFileNpy = filePath2Npy(eachFilePath)
        fileImgList.append(eachFileNpy)




    graph = tf.get_default_graph()



    latent_op = graph.get_tensor_by_name('latentLayer/latentLayer_sig:0')

    allStep = train_batches_per_epoch = int(np.floor(len(fileImgList) / batchSize)) + 1

    print('feed forwarding...')

    with tf.Session() as sess:
        saver.restore(sess, modelPath)
        count = 0

        for step in range(allStep):

            batchList = []
            overBatch = 0

            if(step % 20 == 0):
                print('for ' + str(step) + ' feedforwarding...')

            for k in range(batchSize):
                if(count < len(fileImgList)):
                    batchList.append(fileImgList[count])
                    count=count+1
                else:
                    batchList.append(fileImgList[0])
                    overBatch = overBatch + 1

            batchListNpy = np.asarray(batchList)

            bitRepresentation = sess.run(latent_op, feed_dict={'input_network:0': batchListNpy, 'keep_prob:0': 1.0})


            for j in range(len(bitRepresentation) - overBatch):
                for k in range(len(bitRepresentation[j])):
                    probValue = bitRepresentation[j][k]
                    if(probValue > 0.5):
                        bitRepresentation[j][k] = 1.0
                    else:
                        bitRepresentation[j][k] = 0.0

                bitList.append(bitRepresentation[j])

    allWriteContents = []

    for k in range(len(bitList)):
        singleFilePath = filePathList[k]
        singleLabel = labelList[k]
        singleBit = bitList[k]
        singleBitStr = []
        singleBitSize = np.shape(singleBit)[0]

        singleLineContents = singleFilePath + ',' + singleLabel

        for j in range(singleBitSize):
            singleLineContents = singleLineContents + ',' + str(singleBit[j])

        allWriteContents.append(singleLineContents)

    outputPath = os.path.join(outputParentPath, additionalPrefix + '_' + modelPath.split('/')[-1] + "_isTrain_" + str(isTrain) + '_bit_result.csv')
    with open(outputPath, 'w') as writeFile:
        for singleWriteContents in allWriteContents:
            writeFile.write(singleWriteContents + '\n')

    print('done')






def extractSPDHandSave(modelPath='', inputScheme='cropped'):

    num_classes = 6
    batchSize = 32

    # placeholder for input image data

    saver = tf.train.import_meta_graph(modelPath + '.meta')

    if(inputScheme == 'cropped'):
        trainImg_path = '../dataNpyFiles/bbox_cropped_train_img.npy'
        trainLabel_path = '../dataNpyFiles/bbox_cropped_train_label.npy'
        testImg_path = '../dataNpyFiles/bbox_cropped_test_img.npy'
        testLabel_path = '../dataNpyFiles/bbox_cropped_test_label.npy'
    elif(inputScheme =='activated'):
        trainImg_path = '../dataNpyFiles/bbox_activated_train_img.npy'
        trainLabel_path = '../dataNpyFiles/bbox_activated_train_label.npy'
        testImg_path = '../dataNpyFiles/bbox_activated_test_img.npy'
        testLabel_path = '../dataNpyFiles/bbox_activated_test_label.npy'
    elif(inputScheme == 'origin'):
        trainImg_path = '../dataNpyFiles/origin_train_img.npy'
        trainLabel_path = '../dataNpyFiles/origin_train_label.npy'
        testImg_path = '../dataNpyFiles/origin_test_img.npy'
        testLabel_path = '../dataNpyFiles/origin_test_label.npy'
    else:
        print('Wrong with param...')
        return

    print('load train data...')
    trainData = DataManager(imgNpyPath=trainImg_path, labelNpyPath=trainLabel_path, batchSize=batchSize,
                                        classNumber=num_classes)
    print('load test data...')
    testData = DataManager(imgNpyPath=testImg_path, labelNpyPath=testLabel_path, batchSize=batchSize,
                                       classNumber=num_classes)

    trainDataSize = np.shape(trainData.label_data)[0]
    testDataSize = np.shape(testData.label_data)[0]





    train_batches_per_epoch = int(np.floor(trainDataSize / batchSize))
    val_batches_per_epoch = int(np.floor(testDataSize / batchSize))




    train_bit_representation = []
    train_label_representation = []

    test_bit_representation = []
    test_label_representation = []



    graph = tf.get_default_graph()

    # load for all operation restored

    latent_op = graph.get_tensor_by_name('latentLayer/latentLayer_sig:0')



    # x = tf.placeholder(tf.float32, [32, 227, 227, 3], name='Placeholder_4')

    for op in graph.get_operations():
        print(op.name)




    with tf.Session() as sess:
        saver.restore(sess, modelPath)

        print('bit generation for train...')

        # for extract bit of training data
        print('trainData initializer on GPU')
        q = trainData.getTrueData()
        sess.run(trainData.getInitializer(), feed_dict={trainData.x: q[0], trainData.y: q[1]})

        for step in range(train_batches_per_epoch):
            if(step % 20 == 0):
                print('For ' + str(step))
            img_batch, label_batch = sess.run(trainData.getNextBatchPlaceholder())

            # bitRepresentation = sess.run('latentLayer/Sigmoid:0', feed_dict={'Placeholder_4:0': img_batch, 'Placeholder_6:0': 1.0})
            bitRepresentation = sess.run(latent_op, feed_dict={'input_network:0': img_batch, 'keep_prob:0': 1.0})

            train_label_representation.append(label_batch)
            train_bit_representation.append(bitRepresentation)

        # for extract bit of testing data
        print('bit generation for test...')

        print('testData initializer on GPU')
        p = testData.getTrueData()
        sess.run(testData.getInitializer(), feed_dict={testData.x: p[0], testData.y: p[1]})

        for step in range(val_batches_per_epoch):
            if (step % 20 == 0):
                print('For ' + str(step))
            # Is it right??
            img_batch, label_batch = sess.run(testData.getNextBatchPlaceholder())

            bitRepresentation = sess.run(latent_op, feed_dict={'input_network:0': img_batch, 'keep_prob:0': 1.0})

            test_label_representation.append(label_batch)
            test_bit_representation.append(bitRepresentation)

        train_bit_representation_npy = np.asarray(train_bit_representation)
        test_bit_representation_npy = np.asarray(test_bit_representation)

        train_label_representation_npy = np.asarray(train_label_representation)
        test_label_representation_npy = np.asarray(test_label_representation)

        train_save_dic = {'bit': train_bit_representation_npy, 'label': train_label_representation_npy}
        test_save_dic = {'bit': test_bit_representation_npy, 'label': test_label_representation_npy}

        print('save train bit...')
        np.save(modelPath.split('/')[-1] + '_train_bit.npy', train_save_dic)
        print('save test bit...')
        np.save(modelPath.split('/')[-1] + '_test_bit_npy', test_save_dic)

        print('done')


def testAllTrainedModelShell():
    outputPath = '../bitStore/180511'
    parentPath = '../modelStore/ASAN/modelTrainedPath'
    # For cropped

    inputSchemeVariation = ['Activated', 'Cropped', 'Origin']
    trainVariation = ['normal', 'onlySC']
    bitVariation = ['16bit', '32bit', '48bit']

    for inputScheme in inputSchemeVariation:
        for trainScheme in trainVariation:
            for bitScheme in bitVariation:
                leafParentAbsPath = os.path.join(parentPath, inputScheme, trainScheme, bitScheme)
                modelList = os.listdir(leafParentAbsPath)

                modelList.sort()

                prevModel = ''

                for eachModel in modelList:

                    modelName = '.'.join(eachModel.split('.')[0:-1])
                    print('process for ' + modelName)

                    if(prevModel != modelName):
                        prevModel = modelName
                    else:
                        continue

                    eachModelAbsPath = os.path.join(leafParentAbsPath, modelName)
                    testAllTrainedModel(inputMode=inputScheme, modelPath=eachModelAbsPath, outputPath=outputPath)


def testAllTrainedModel(inputMode, modelPath, outputPath):

    if(inputMode == 'Origin'):
        labelFilePathSearch = '../exp_jh/origin_search.label'
        labelFilePathQuery = '../exp_jh/origin_query.label'
    elif(inputMode == 'Cropped'):
        labelFilePathSearch = '../exp_jh/bbox_search.label'
        labelFilePathQuery = '../exp_jh/bbox_query.label'
    elif(inputMode == 'Activated'):
        labelFilePathSearch = '../exp_jh/activated_search.label'
        labelFilePathQuery = '../exp_jh/activated_query.label'
    else:
        print('wrong with input mode...')
        return

    print('extract for train...')
    extractSPDH(modelPath=modelPath, labelPath=labelFilePathSearch, isTrain=True, additionalPrefix='JH', outputParentPath=outputPath)

    print('extract for test...')
    extractSPDH(modelPath=modelPath, labelPath=labelFilePathQuery, isTrain=False, additionalPrefix='JH', outputParentPath=outputPath)


testAllTrainedModelShell()


print('done')








