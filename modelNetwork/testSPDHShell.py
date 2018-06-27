import argparse

from testSPDH import extractSPDH


def main(args):
    useEpochNum = args.useEpochNum
    additionalPrefix = args.additionalPrefix
    modelPath = args.modelPath + str(useEpochNum) + '.ckpt'
    labelpath = args.labelPath
    isTrain = args.isTrain

    extractSPDH(modelPath=modelPath, labelPath=labelpath, isTrain=isTrain, additionalPrefix=additionalPrefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--useEpochNum',
        type=str,
        help='Epoch number to use in trained model',
        required=True)

    parser.add_argument(
        '--additionalPrefix',
        type=str,
        help='addition prefix',
        required=True)

    parser.add_argument(
        '--modelPath',
        type=str,
        help='model path',
        required=True)
    parser.add_argument(
        '--labelPath',
        type=str,
        help='input label path',
        required=True)
    parser.add_argument(
        '--isTrain',
        type=bool,
        help='is Train or Test',
        required=True)

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)


'''
# test train model with args

from testTrainModel import train_SPHash

import argparse


# train_SPHash(inputType="Origin", bitSize=16, num_epochs=50)


def main(args):
    inputType = args.inputType
    bitSize = args.bitSize
    num_epochs = args.num_epochs

    print('//////////////////////////////////////////////////////////////////////////////')
    print('train condition for ' + inputType + '_' + str(bitSize) + '_' + str(num_epochs))
    print('//////////////////////////////////////////////////////////////////////////////')

    train_SPHash(inputType=inputType, bitSize=bitSize, num_epochs=num_epochs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--inputType',
      type=str,
      help='Input type for training(Origin, Activated, Cropped)',
      required=True)
  parser.add_argument(
      '--bitSize',
      type=int,
      help='Bit Size(16, 32, 48)',
      required=True)
  parser.add_argument(
      '--num_epochs',
      type=int,
      help='Number of epochs',
      required = True)

  FLAGS, unparsed = parser.parse_known_args()

  main(FLAGS)




# train_SPHash(inputType="Origin", bitSize=16, num_epochs=50)
'''

epochNumber = 26
epochOriginNumber = 96

labelPathJH = ['../exp_jh/bbox_search.label', '../exp_jh/bbox_query.label',
               '../exp_jh/activated_search.label', '../exp_jh/activated_query.label',
               '../exp_jh/origin_search.label', '../exp_jh/origin_query.label']




extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHCropped_16model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[0],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHCropped_16model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[1],
            isTrain=False, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHCropped_32model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[0],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHCropped_32model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[1],
            isTrain=False, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHCropped_48model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[0],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHCropped_48model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[1],
            isTrain=False, additionalPrefix='JH')







extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHActivated_16model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[2],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHActivated_16model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[3],
            isTrain=False, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHActivated_32model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[2],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHActivated_32model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[3],
            isTrain=False, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHActivated_48model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[2],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHActivated_48model_epoch' + str(epochNumber) + '.ckpt',
            labelPath=labelPathJH[3],
            isTrain=False, additionalPrefix='JH')







extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHOrigin_16model_epoch' + str(epochOriginNumber) + '.ckpt',
            labelPath=labelPathJH[4],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHOrigin_16model_epoch' + str(epochOriginNumber) + '.ckpt',
            labelPath=labelPathJH[5],
            isTrain=False, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHOrigin_32model_epoch' + str(epochOriginNumber) + '.ckpt',
            labelPath=labelPathJH[4],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHOrigin_32model_epoch' + str(epochOriginNumber) + '.ckpt',
            labelPath=labelPathJH[5],
            isTrain=False, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHOrigin_48model_epoch' + str(epochOriginNumber) + '.ckpt',
            labelPath=labelPathJH[4],
            isTrain=True, additionalPrefix='JH')

extractSPDH(modelPath='../modelStore/ASAN/modelTrainedPath/SPDHOrigin_48model_epoch' + str(epochOriginNumber) + '.ckpt',
            labelPath=labelPathJH[5],
            isTrain=False, additionalPrefix='JH')


print('done')








