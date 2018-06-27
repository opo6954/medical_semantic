from testTrainModel import train_SPHash_cifar10


# def train_SPHash_cifar10(modelSavedPath = '../modelStore/ASAN', trainLabel_path='', testLabel_path='', bitSize=16, num_epochs=30, weightSM=1.0, weightApp=1.0, weightFire=1.0):

modelSavedPath = '../modelStore/ASAN/180530'
trainLabel_path = '../exp_cifar10/labelfile/cifar10_train.label'
testLabel_path = '../exp_cifar10/labelfile/cifar10_test.label'
num_epochs = 50




train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=16, weightSM=1.0, weightApp=1.0, weightFire=1.0 )
train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=16, weightSM=1.0, weightApp=0.0, weightFire=0.0 )
train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=16, weightSM=0.5, weightApp=0.3, weightFire=0.2)

train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=32, weightSM=1.0, weightApp=1.0, weightFire=1.0 )

train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=32, weightSM=1.0, weightApp=0.0, weightFire=0.0 )

train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=32, weightSM=0.5, weightApp=0.3, weightFire=0.2 )

train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=48,weightSM=1.0, weightApp=1.0, weightFire=1.0 )

train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=48, weightSM=1.0, weightApp=0.0, weightFire=0.0 )

train_SPHash_cifar10(modelSavedPath=modelSavedPath,
                     trainLabel_path=trainLabel_path,
                     testLabel_path=testLabel_path,
                    num_epochs=num_epochs,
                     bitSize=48,weightSM=0.5, weightApp=0.3, weightFire=0.2 )







