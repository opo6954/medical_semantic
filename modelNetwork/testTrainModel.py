
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import datetime
import modelClass
sys.path.insert(0, '../dataProcessor')

import dataManager
import dataManagerFromDisk





from SPHash_tf import SPDH_tf

import alexnet
import tfWrapper


savor_max_to_keep = 5
model_save_step = 5

def train_SPHash(modelSavedPath = '../largeData/model', inputType='Cropped', bitSize=16, num_epochs=30, weightSM = 1.0, weightApp = 1.0, weightFire = 1.0):
    tf.reset_default_graph()


    prefix = 'JH_SPDH' + inputType + '_' + str(bitSize) + '_wSM_' + str(weightSM) + '_wApp_' + str(weightApp) + '_wFire_' + str(weightFire)

    batchSize = 32
    dropout_rate = 0.5
    display_step = 20

    # For ASAN Data

    num_classes = 6

    # filewriter_path = '../modelStore/ASAN/tensorboardPath'
    # checkpoint_path = '../modelStore/ASAN/modelTrainedPath'

    filewriter_path = modelSavedPath + '/tensorboardPath'
    checkpoint_path = modelSavedPath + '/modelTrainedPath'

    if(not os.path.exists(filewriter_path)):
        os.mkdir(filewriter_path)
    if(not os.path.exists(checkpoint_path)):
        os.mkdir(checkpoint_path)


    if (inputType == 'Cropped'):
        trainImg_path = '../largeData/data/exp_jh/bbox_search_img.npy'
        trainLabel_path = '../largeData/data/exp_jh/bbox_search_label.npy'
        testImg_path = '../largeData/data/exp_jh/bbox_query_img.npy'
        testLabel_path = '../largeData/data/exp_jh/bbox_query_label.npy'
    elif (inputType == 'Activated'):
        trainImg_path = '../largeData/data/exp_jh/activated_search_img.npy'
        trainLabel_path = '../largeData/data/exp_jh/activated_search_label.npy'
        testImg_path = '../largeData/data/exp_jh/activated_query_img.npy'
        testLabel_path = '../largeData/data/exp_jh/activated_query_label.npy'
    elif (inputType == 'Origin'):
        trainImg_path = '../largeData/data/exp_jh/origin_search_img.npy'
        trainLabel_path = '../largeData/data/exp_jh/origin_search_label.npy'
        testImg_path = '../largeData/data/exp_jh/origin_query_img.npy'
        testLabel_path = '../largeData/data/./exp_jh/origin_query_label.npy'
    else:
        print ('No available option found for ' + inputType)
        return

    print('load train data...')
    trainData = dataManager.DataManager(imgNpyPath=trainImg_path, labelNpyPath=trainLabel_path, batchSize=batchSize, classNumber=num_classes)
    print('load test data...')
    testData = dataManager.DataManager(imgNpyPath=testImg_path, labelNpyPath=testLabel_path, batchSize=batchSize, classNumber=num_classes)


    trainDataSize = np.shape(trainData.label_data)[0]
    testDataSize = np.shape(testData.label_data)[0]

    x = tf.placeholder(tf.float32, [batchSize, 227, 227, 3], name='input_network')
    y = tf.placeholder(tf.float32, [batchSize, num_classes], name='label_network')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')



    model = SPDH_tf(bitSize=bitSize, x_shape=x, y_shape=y, keep_prob=keep_prob, num_classes=num_classes,
                    skip_layer=['fc8', 'fc7', 'fc6'], weights_path='../largeData/modelAlexnet/bvlc_alexnet.npy',
                    batchSize=batchSize, weightSC=weightSM, weightApp=weightApp, weightFire=weightFire)


    model.build_basic_graph()
    model.design_loss()

    writer = tf.summary.FileWriter(filewriter_path)
    saver = tf.train.Saver(max_to_keep=savor_max_to_keep)

    train_batches_per_epoch = int(np.floor(trainDataSize / batchSize))
    val_batches_per_epoch = int(np.floor(testDataSize / batchSize))



    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        writer.add_graph(sess.graph)

        model.load_initial_weights(sess)

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(),
                                                          filewriter_path))





        # Loop over number of epochs
        for epoch in range(num_epochs):

            print('trainData initializer on GPU')
            q = trainData.getTrueData()
            sess.run(trainData.getInitializer(), feed_dict={trainData.x: q[0], trainData.y: q[1]})

            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))



            for step in range(train_batches_per_epoch):

                input_value, label_value = sess.run(trainData.getNextBatchPlaceholder())

                # And run the training op
                # trainLoss, _ = sess.run([loss, optimizer], feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})

                # for TEST
                p = sess.run(model.latentLayer, feed_dict={model.x: input_value, model.keep_prob: 1.0})





                lossTotal, lossSC, lossApp, lossFire, _ = sess.run([model.loss_total, model.loss_sc, model.loss_app, model.loss_fire, model.train_op],
                                        feed_dict={model.x: input_value, model.y: label_value, model.keep_prob: dropout_rate})
                # trainLoss = sess.run(train_op, feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})

                # Generate summary with the current
                #
                # cat batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(model.merged_summary, feed_dict={model.x: input_value,
                                                            model.y: label_value,
                                                            model.keep_prob: 1.})
                    print('At ' + str(step) + ', train loss: ' + str(lossTotal) + ', train SC: ' + str(lossSC) + ', train App: ' + str(lossApp) + ', trainFire: ' + str(lossFire))

                    writer.add_summary(s, epoch * train_batches_per_epoch + step)


            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.datetime.now()))

            test_acc = 0.
            test_count = 0

            print('testData initializer on GPU')
            p = testData.getTrueData()
            sess.run(testData.getInitializer(), feed_dict={testData.x: p[0], testData.y: p[1]})



            for _ in range(val_batches_per_epoch):
                img_batch, label_batch = sess.run(testData.getNextBatchPlaceholder())

                acc = sess.run(model.accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count





            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            # save checkpoint of the model
            if(epoch % model_save_step == 0):
                checkpoint_name = os.path.join(checkpoint_path, prefix + 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_name))



train_SPHash(inputType="Cropped", bitSize=32, num_epochs=10)

'''
train_SPHash(inputType="Cropped", bitSize=32, num_epochs=10)
train_SPHash(inputType="Activated", bitSize=32, num_epochs=10)
train_SPHash(inputType="Origin", bitSize=32, num_epochs=20)


train_SPHash(inputType="Cropped", bitSize=48, num_epochs=20)
train_SPHash(inputType="Activated", bitSize=48, num_epochs=20)
train_SPHash(inputType="Origin", bitSize=48, num_epochs=500)

train_SPHash(inputType="Cropped", bitSize=16, num_epochs=20)
train_SPHash(inputType="Activated", bitSize=16, num_epochs=20)
train_SPHash(inputType="Origin", bitSize=16, num_epochs=50)
'''
# test_pretrained(isNIH=False)