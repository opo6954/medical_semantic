from testTrainModel import train_SPHash_givenNpy
from testTrainModel import train_SPHash_loadDisk

'''
train_SPHash_givenNpy(num_classes=10, prefix='cifar10', modelSavedPath = '../largeData/model',
                      inputTrainImgNpyFile='../largeData/cifar10/exp_cifar10/npyfile/cifar10_train_img.npy',
                      inputTrainLabelNpyFile='../largeData/cifar10/exp_cifar10/npyfile/cifar10_train_label.npy',
                      inputTestImgNpyFile='../largeData/cifar10/exp_cifar10/npyfile/cifar10_test_img.npy',
                      inputTestLabelNpyFile='../largeData/cifar10/exp_cifar10/npyfile/cifar10_test_label.npy',
                      bitSize=16, num_epochs=30, weightSM=1.0, weightApp=1.0, weightFire=1.0)
'''

train_SPHash_loadDisk(num_classes=10, prefix='cifar10', modelSavedPath='../largeData/model',
                       inputTrainLabelFile='../largeData/cifar10/exp_cifar10/labelfile/cifar10_train.label',
                       inputTestLabelFile='../largeData/cifar10/exp_cifar10/labelfile/cifar10_test.label',
                       bitSize=16, num_epochs=50, weightSM=1.0, weightApp=1.0, weightFire=1.0)

train_SPHash_loadDisk(num_classes=10, prefix='cifar10', modelSavedPath='../largeData/model',
                       inputTrainLabelFile='../largeData/cifar10/exp_cifar10/labelfile/cifar10_train.label',
                       inputTestLabelFile='../largeData/cifar10/exp_cifar10/labelfile/cifar10_test.label',
                       bitSize=32, num_epochs=50, weightSM=1.0, weightApp=1.0, weightFire=1.0)

train_SPHash_loadDisk(num_classes=10, prefix='cifar10', modelSavedPath='../largeData/model',
                       inputTrainLabelFile='../largeData/cifar10/exp_cifar10/labelfile/cifar10_train.label',
                       inputTestLabelFile='../largeData/cifar10/exp_cifar10/labelfile/cifar10_test.label',
                       bitSize=48, num_epochs=50, weightSM=1.0, weightApp=1.0, weightFire=1.0)





